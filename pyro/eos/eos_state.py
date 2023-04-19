class eos_t:

    def __init__(self, eos_inputs, var1, var2):

        self.eos_inputs = eos_inputs

        if self.eos_inputs == 'rT':
            self.rho = var1
            self.T = var2

        elif self.eos_inputs == 'rp':
            self.rho = var1
            self.pres = var2

        elif self.eos_inputs == 're':
            self.rho = var1
            self.eint = var2

        elif self.eos_inputs == 'pT':
            self.rho = var1
            self.T = var2

        elif self.eos_inputs == 'pe':
            self.pres = var1
            self.eint = var2

        elif self.eos_inputs == 'Te':
            self.T = var1
            self.eint = var2

        else:
            raise NotImplementedError

    def gamma(self, gamma=1.40):

    def helmholtz_table(self, file):

    def helmholtz(self, helmholtz_table):
