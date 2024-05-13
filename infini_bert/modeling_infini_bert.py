## https://chat.openai.com/share/f5765f7a-b9e6-4280-825e-80a9f2b36974
import math

import torch
from torch import nn

class BertInfiniModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel(config)  # TODO: wut!!! Assuming BertModel integrates BertInfiniSelfAttention
        self.memory_keys = None
        self.memory_values = None

    def reset_memory(self, batch_size):
        # Resets memory for each new sequence in the batch
        self.memory_keys = [torch.zeros(self.config.hidden_size, self.config.hidden_size)
                            for _ in range(batch_size)]
        self.memory_values = [torch.zeros(self.config.hidden_size, self.config.hidden_size)
                              for _ in range(batch_size)]

    def forward(self, input_ids, attention_mask=None):
        batch_size = input_ids.size(0)
        self.reset_memory(batch_size)

        # Process each segment of input_ids here
        # pseudo-code:
        # for each segment in input_ids:
        #    output, memory_keys, memory_values = self.bert(segment, self.memory_keys, self.memory_values)
        #    # Update memory after processing each segment
        #    self.memory_keys, self.memory_values = update_memory(memory_keys, memory_values)

        return output

    def update_memory(self, old_memory_keys, old_memory_values, new_memory_keys, new_memory_values):
        # Logic to update memory based on old and new values
        return new_memory_keys, new_memory_values



class BertInfiniSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Gate and memory parameters
        self.gate = nn.Parameter(torch.zeros(1, self.num_attention_heads, 1, 1))
        self.memory_keys = nn.ParameterList([nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))
                                             for _ in range(self.num_attention_heads)])
        self.memory_values = nn.ParameterList([nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))
                                               for _ in range(self.num_attention_heads)])

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Apply attention on current input and memory
        key_layer_combined = torch.cat([self.memory_keys[i] @ key_layer[:, i] for i in range(self.num_attention_heads)], dim=1)
        value_layer_combined = torch.cat([self.memory_values[i] @ value_layer[:, i] for i in range(self.num_attention_heads)], dim=1)

        attention_scores = torch.matmul(query_layer, key_layer_combined.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer_combined)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Combine using gates
        gate_scores = torch.sigmoid(self.gate)
        combined_context_layer = gate_scores * context_layer + (1 - gate_scores) * self.retrieve_from_memory(query_layer)

        return combined_context_layer

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def retrieve_from_memory(self, query_layer):
        # Implement memory retrieval logic similar to LlamaInfiniAttention
        pass

    def update_memory(self, key_layer, value_layer):
        # Implement memory update logic similar to LlamaInfiniAttention
        pass


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertInfiniSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(attention_output, hidden_states)
        return attention_output
