class Simulations: 
    """
    A class to perform simulations for different topic models with different settings
    """
    def __init__(self, settings, documents):
        self.settings = settings 
    #____________________________________________________________________________
    #  Init Topic Models
    #____________________________________________________________________________
    def init_lda(self):
        pass
    def init_dmr(self):
        pass
    def init_ctm(self):
        pass
    def init_stm(self):
        pass
       
    #____________________________________________________________________________
    # Fit Topic Models
    #____________________________________________________________________________
    def fit_lda(self):
        pass
    def fit_dmr(self):
        pass
    def fit_ctm(self):
        pass
    def fit_stm(self):
        pass
    def fit_topic_models(self):
        self.fit_lda()
        self.fit_dmr()
        self.fit_ctm()
        self.fit_stm()

    #____________________________________________________________________________
    # Evaluate Topic Models
    #____________________________________________________________________________
    def lda_perplexity(self):
        self.compute_perplexity(self.lda)
    
    def ctm_perplexity(self):
        self.compute_perplexity(self.ctm)
    
    def stm_perplexity(self):
        self.compute_perplexity(self.stm)
    
    def compute_perplexity(self): 
        pass
    
    def perplexity_plot(self):
        """
        Parameters: 
        lda_perplexity
        ctm_perplexity
        stm_perplexity
        """ 
        pass
    
    def frex_plot(self):
        pass
    def computation_plot(self):
        pass

    def evaluate(self):
        self.perplexity_plot()
        self.computation_plot()
        self.frex_plot()

