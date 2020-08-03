class LNPDF():
    def log_density(self, x):
        raise NotImplementedError

    def gradient_log_density(self, x):
        raise NotImplementedError

    def get_num_dimensions(self):
        raise NotImplementedError

    def can_sample(self):
        return False

    def sample(self, n):
        raise NotImplementedError
