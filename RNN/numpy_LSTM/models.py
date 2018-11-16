import numpy as np

class rnn:
    def __init__(self, num_units, vocab_size, seq_len):
        self.tables = None

        self.seq_len = seq_len
        self.num_units = num_units
        self.vocab_size = vocab_size

    def _init_model(self, n_features, seq_len, num_units):
        self.tables = []
        self.u = np.zeros([num_units, n_features])
        self.w = np.zeros([num_units, num_units])
        self.v = np.zeros([n_features, num_units])

        # s = f(ux + ws)
        # o = vs
        for _ in xrange(seq_len):
            s = np.zeros([n_features, 1])
            self.tables.append(s)

    def _tanh(self, x):
        return np.tanh(x)

    def _softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

    def _cross_entropy(self, y, label):
        return np.matmul(-label, np.log(y))

    def _step(self, x, idx):
        prev_state = None
        if idx == 0:
            prev_state = np.zeros_like(self.tables[idx])
        else:
            prev_state = self.tables[idx - 1]

        new_state = self._tanh(np.matmul(self.u, x) + np.matmul(self.w, prev_state))
        o = self._softmax(np.matmul(self.v, new_state))

        self.tables[idx] = new_state

        return o

    def _bptt(self, loss):
        pass

    def _forward(self, x):
        n_features = x.shape[-1]
        outputs = np.zeros([self.seq_len, n_features])

        for i in xrange(self.seq_len):
            o = self._step(x[i], i)
            outputs[i] = o
        
        return outputs

    def fit(self, x, y, batch_size=32):
        assert (len(x.shape) == 3)
        assert (x.shape[1] == self.seq_len)

        if self.tables is None:
            n_features = x.shape[-1]
            self._init_model(n_features, self.seq_len, self.num_units)

        n_batches = x.shape[0] / batch_size + 1 if x.shape[0] % batch_size == 0 else 0
        for batch_idx in xrange(n_batches):
            batch = x[batch_idx]

            outputs = self._forward(batch)
            loss = self._cross_entropy(outputs, y[batch_idx])
            loss /= n_batches

            self._bptt(loss)

    def predict(self, x):
        return self._forward(x)

if __name__ == "__main__":
    pass