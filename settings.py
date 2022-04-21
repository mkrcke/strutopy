K = 10 #settings.dim
V = len(dictionary) #settings.dim
N = len(documents) #settings.dim

interactions = True #settings.kappa
verbose = True

init_type = "Random" #settings.init
ngroups = 1 #settings.ngroups
max_em_its = 15 #settings.convergence
emtol = 1e-5 #settings.convergence

settings = {
    'dim':{
        'K': K, #number of topics
        'V' :V, #number of words
        'A' : A, #dimension of topical content
        'N' : N,
        'wcounts':V
    },
    'kappa':{
        'interactions':True,
        'fixedintercept': True,
        'contrats': False,
        'mstep': {'tol':0.01, 'maxit':5}},
    'tau':{
        'mode': np.nan,
        'tol': 1e-5,
        'enet':1,
        'nlambda':250,
        'lambda.min.ratio':.001,
        'ic.k':2,
        'maxit':1e4},
    'init':{
        'mode':init_type, 
        'nits':20,
        'burnin':25,
        'alpha':50/K,
        'eta':.01,
        's':.05,
        'p':3000},
    'convergence':{
        'max.em.its':max_em_its,
        'em.converge.thresh':emtol,
        'allow.neg.change':True,},
    'covariates':{
        'X':xmat,
        'betaindex':betaindex,
        'yvarlevels':yvarlevels,
        'formula': prevalence,},
    'gamma':{
        'mode':np.nan,
        'prior':np.nan,
        'enet':1, 
        'ic.k':2,
        'maxits':1000,},
    'sigma':{
        #'prior':sigma_prior,
        'ngroups':ngroups,},
}