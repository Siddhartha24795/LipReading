import os

from allennlp.nn.util import masked_log_softmax, masked_softmax, sort_batch_by_length
import torch
import torch.nn as nn

from src.data.data_loader import BOS, PAD

_ALLOWED_RNN_TYPES = {'LSTM', 'GRU', 'RNN'}
_ALLOWED_FRAME_PROCESSING = {'flatten'}
_ALLOWED_ATTENTION_TYPES = {'none', 'dot', 'general', '1_layer_nn', 'concat'}

class VideoEncoder(nn.Module):
    def __init__(self, frame_dim, hidden_size, frame_processing='flatten',
                 rnn_type='LSTM', num_layers=1, bidirectional=True, rnn_dropout=0,
                 enable_ctc=False, vocab_size=-1, char2idx=None, device="cpu"):
        """
        When enable_ctc=True, vocab_size and char2idx must be provided
        vocab_size includes all the special tokens
        """
        super(VideoEncoder, self).__init__()
        assert frame_processing in _ALLOWED_FRAME_PROCESSING
        assert rnn_type in _ALLOWED_RNN_TYPES
        if enable_ctc:
            assert vocab_size > 0 and char2idx is not None

        self.frame_dim = frame_dim
        self.hidden_size = hidden_size
        self.frame_processing = frame_processing
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_dropout = rnn_dropout
        self.enable_ctc = enable_ctc
        self.best_error = 1
        if self.enable_ctc:
            self.vocab_size = vocab_size
            self.adj_vocab_size = self.vocab_size + 1
            self.char2idx = char2idx
            self.num_dirs = 2 if self.bidirectional else 1

            # Due to the stupid restriction of PyTorch CTC + CuDNN, we must reserve idx=0 for blank symbol
            self.output_mask = torch.ones(self.adj_vocab_size, device=device)
            self.output_mask[self.char2idx[PAD] + 1] = 0
            self.output_mask[self.char2idx[BOS] + 1] = 0

        self.rnn = getattr(nn, self.rnn_type)(self.frame_dim, self.hidden_size,
                                              num_layers=self.num_layers, bidirectional=self.bidirectional,
                                              batch_first=True, dropout=self.rnn_dropout)
        if self.enable_ctc:
            self.output_proj = nn.Linear(self.num_dirs * self.hidden_size, self.adj_vocab_size)

    def forward(self,
                frames: torch.FloatTensor,
                frame_lens: torch.LongTensor):
        """
        frames: (batch_size, seq_len, num_lmks, lmk_dim)
        frame_lens: (batch_size, )
        """
        if self.frame_processing == 'flatten':
            frames = frames.reshape(frames.shape[0], frames.shape[1], -1)

        # Reverse sorts the batch by unpadded seq_len.
        (sorted_frames, sorted_frame_lens,
            restoration_indices, _) = sort_batch_by_length(frames, frame_lens)

        # Returns a PackedSequence.
        packed_frames = nn.utils.rnn.pack_padded_sequence(sorted_frames,
                            sorted_frame_lens.data.cpu().numpy() if sorted_frame_lens.is_cuda else sorted_frame_lens.data.numpy(),
                            batch_first=True)

        # Encoder: feed frames to the model, output hidden states.
        # final_state: (num_layers * num_dir, batch_size, hidden_size) (*2 if LSTM)
        packed_hidden_states, final_state = self.rnn(packed_frames)

        # Unpack encoding, the hidden states, a Tensor.
        # (batch_size, seq_len, num_dir * hidden_size)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(packed_hidden_states, batch_first=True)

        # (num_layers, batch_size, hidden_size * num_dir) (*2 if LSTM)
        if self.bidirectional:
            final_state = self._cat_directions(final_state)

        hidden_states = hidden_states.index_select(0, restoration_indices)
        if isinstance(final_state, tuple):  # LSTM
            final_state = (final_state[0].index_select(1, restoration_indices),
                           final_state[1].index_select(1, restoration_indices))
        else:
            final_state = final_state.index_select(1, restoration_indices)

        if self.enable_ctc:
            output_logits = self.output_proj(hidden_states)
            output_log_probs = masked_log_softmax(output_logits, self.output_mask.expand(output_logits.shape[0], self.adj_vocab_size), dim=-1)
            return output_log_probs, hidden_states, final_state
        else:
            return hidden_states, final_state

    def _cat_directions(self, final_state):
        """
        final_state must come from a bidirectional RNN
        (num_layers * num_dir, batch_size, hidden_size) -->
        (num_layers, batch_size, hidden_size * num_dir)
        """
        def _cat(s):
            return torch.cat([s[0:s.shape[0]:2], s[1:s.shape[0]:2]], dim=2)

        if isinstance(final_state, tuple):  # LSTM
            final_state = tuple(_cat(s) for s in final_state)
        else:
            final_state = _cat(final_state)

        return final_state

    def save_best_model(self, error, file_path):
        if error < self.best_error:

            self.best_error = error
            folder = os.path.dirname(file_path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(self.state_dict(), file_path)
            print("\tSaving best error '{}' to '{}'".format(self.best_error, file_path))

class CharDecodingStep(nn.Module):
    def __init__(self, encoder: VideoEncoder, char_dim, vocab_size, char2idx, rnn_dropout=0, attention_type='none', attn_hidden_size=-1, device="cpu"):
        """
        vocab_size includes all the special tokens
        """
        super(CharDecodingStep, self).__init__()

        assert attention_type in _ALLOWED_ATTENTION_TYPES
        if attention_type == 'concat':
            assert attn_hidden_size > 0
        self.hidden_size = encoder.hidden_size * (2 if encoder.bidirectional else 1)
        self.rnn_type = encoder.rnn_type
        self.num_layers = encoder.num_layers
        self.rnn_dropout = rnn_dropout
        self.char_dim = char_dim
        self.vocab_size = vocab_size
        self.char2idx = char2idx
        self.attention_type = attention_type

        self.output_mask = torch.ones(self.vocab_size, device=device)
        self.output_mask[self.char2idx[PAD]] = 0
        self.output_mask[self.char2idx[BOS]] = 0
        self.embedding = nn.Embedding(self.vocab_size, self.char_dim, padding_idx=self.char2idx[PAD])
        self.rnn = getattr(nn, self.rnn_type)(self.char_dim, self.hidden_size,
                                              num_layers=self.num_layers, batch_first=True, dropout=self.rnn_dropout)
        if attention_type == '1_layer_nn':
            self.attn_proj_1_layer_nn = nn.Linear(2 * self.hidden_size, 1)
        elif attention_type == 'general':
            self.attn_proj_general = nn.Linear(self.hidden_size, self.hidden_size)
        elif attention_type == 'concat':
            self.attn_proj_layer1 = nn.Linear(2 * self.hidden_size, attn_hidden_size)
            self.attn_proj_layer2 = nn.Linear(attn_hidden_size, 1)
        self.concat_layer = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.output_proj = nn.Linear(self.hidden_size, self.vocab_size)

        self.best_error = 1

    def forward(self,
                input_: torch.LongTensor,
                previous_state: torch.FloatTensor,
                encoder_lens: torch.LongTensor,
                encoder_hidden_states: torch.FloatTensor):
        """
        input_: (batch_size, )
        previous_state: (num_layers, batch_size, hidden_size)
        encoder_lens: (batch_size, )
        encoder_hidden_states: (batch_size, en_seq_len, hidden_size)
        """
        batch_size = input_.shape[0]
        en_seq_len = encoder_hidden_states.shape[1]

        # (batch_size, en_seq_len)
        encoder_mask = torch.arange(en_seq_len, device=input_.device).expand(batch_size, en_seq_len) < encoder_lens.unsqueeze(dim=1)
        # (batch_size, char_dim)
        embedded_char = self.embedding(input_)
        # (batch_size, seq_len=1, char_dim)
        embedded_char = embedded_char.unsqueeze(dim=1)

        # hidden_state: (batch_size, seq_len=1, hidden_size)
        # final_state: (num_layers, batch_size, hidden_size)
        hidden_state, final_state = self.rnn(embedded_char, previous_state)

        # (batch_size, en_seq_len, hidden_size)
        expanded_hidden_state = hidden_state.expand_as(encoder_hidden_states)

        if self.attention_type == '1_layer_nn':
            # (batch_size, en_seq_len, hidden_size * 2)
            concat_hidden_state = torch.cat([encoder_hidden_states, expanded_hidden_state], dim=2)
            # (batch_size, en_seq_len)
            attn_logits = self.attn_proj_1_layer_nn(concat_hidden_state).squeeze(dim=-1)
        elif self.attention_type == 'general':
            # (batch_size, en_seq_len, 1, hidden_size)
            expanded_hidden_state = expanded_hidden_state.unsqueeze(dim=2)
            # (batch_size, en_seq_len, hidden_size, 1)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=-1)
            # (batch_size, en_seq_len, 1, hidden_size)
            intermediate_step = self.attn_proj_general(expanded_hidden_state)
            # (batch_size * en_seq_len, 1, 1)
            # (batch_size, en_seq_len, 1, hidden_size) x (batch_size, en_seq_len, hidden_size, 1)
            attn_logits = intermediate_step.view(-1, 1, self.hidden_size).bmm(encoder_hidden_states.view(-1, self.hidden_size, 1))
            # (batch_size, en_seq_len)
            attn_logits = attn_logits.squeeze(dim=-1).squeeze(dim=-1).view(batch_size, -1)
            # (batch_size, en_seq_len, hidden_size)
            expanded_hidden_state = expanded_hidden_state.squeeze(dim=2)
            # (batch_size, en_seq_len, hidden_size)
            encoder_hidden_states = encoder_hidden_states.squeeze(dim=-1)
        elif self.attention_type == 'dot':
            # (batch_size, en_seq_len)
            attn_logits = torch.sum(expanded_hidden_state.mul(encoder_hidden_states), dim=-1)
        elif self.attention_type == 'concat':
            # (batch_size, en_seq_len, hidden_size * 2)
            concat_hidden_state = torch.cat([encoder_hidden_states, expanded_hidden_state], dim=2)
            # (batch_size, en_seq_len, attn_hidden_size)
            attn_logits = self.attn_proj_layer1(concat_hidden_state).tanh()
            # (batch_size, en_seq_len)
            attn_logits = self.attn_proj_layer2(attn_logits).squeeze(dim=-1)

        if self.attention_type != 'none':
            # (batch_size, 1, en_seq_len)
            attn_weights = masked_softmax(attn_logits, encoder_mask, dim=-1).unsqueeze(dim=1)
            # (batch_size, hidden_size)
            context = attn_weights.bmm(encoder_hidden_states).squeeze(dim=1)
            # (batch_size, hidden_size)
            new_hidden_state = self.concat_layer(torch.cat([context, hidden_state.squeeze(dim=1)], dim=1))
            new_hidden_state = new_hidden_state.tanh()
            # (batch_size, vocab_size)
            output_logits = self.output_proj(new_hidden_state)
        else:
            output_logits = self.output_proj(hidden_state.squeeze(dim=1))
        output_log_probs = masked_log_softmax(output_logits, self.output_mask.expand(batch_size, self.vocab_size), dim=-1)

        return output_log_probs, final_state

    def save_best_model(self, error, file_path):
        if error < self.best_error:

            self.best_error = error
            folder = os.path.dirname(file_path)
            if not os.path.exists(folder):
                os.makedirs(folder)
            torch.save(self.state_dict(), file_path)
            print("\tSaving best error '{}' to '{}'".format(self.best_error, file_path))
