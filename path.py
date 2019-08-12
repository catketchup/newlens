class file:
    def __init__(self, data_version, datatype):
        self.data_version = data_version
        self.datatype = datatype
        self.file_path = './data/'

    def file_data(self):
        if self.data_version == 1:
            if self.datatype == 'unlensed':
                return self.file_path+'planck_lensing_wp_highL_bestFit_20130627_scalCls.dat'
            if self.datatype == 'lensed':
                return self.file_path+'planck_lensing_wp_highL_bestFit_20130627_lensedCls.dat'

        if self.data_version == 2:
            if self.datatype == 'unlensed':
                return self.file_path + 'cosmo2017_10K_acc3_scalCls.dat'
            if self.datatype == 'lensed':
                return self.file_path+'cosmo2017_10K_acc3_lensedCls.dat'
