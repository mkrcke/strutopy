def random_init(documents, settings): 
      
    K = settings['dim']['K']
    V = settings['dim']['V']
    A = settings['dim']['A']
    N = settings['dim']['N']
    
    #Random initialization
    mu = np.array([0]*(K-1))[:,None]
    sigma = np.zeros(((K-1),(K-1)))
    diag = np.diagonal(sigma, 0)
    diag.setflags(write=True)
    diag.fill(20)
    beta = random.gamma(.1,1, V*K).reshape(K,V)
    beta = (beta / beta.sum(axis=1)[:,None])
    lambd = np.zeros((N, (K-1)))
    
    #turn beta into a list and assign it for each aspect
    beta = [beta, beta] # FOR A=2
    kappa_initialized = init_kappa(documents, K, V, A, interactions=settings['kappa']['interactions'])
    
    #create model object
    model = {'mu':mu, 'sigma':sigma, 'beta': beta, 'lambda': lambd, 'kappa':kappa_initialized}
    
    return(model)

def init_kappa(documents, K, V, A, interactions): 
    # read in documents and vocab
    flat_documents = [item for sublist in documents for item in sublist]
    m = []

    total_sum = sum(n for _, n in flat_documents)

    for elem in flat_documents: 
        m.append(elem[1] / total_sum)

    m = np.log(m) - np.log(np.mean(m)) #logit of m


    #Defining parameters
    aspectmod = A > 1 # if there is more than one topical content variable
    if(aspectmod):
        interact = interactions # allow for the choice to interact
    else:
        interact = FALSE

    #Create the parameters object
    parLength = K + A * aspectmod + (K*A)*interact

    #create covariates. one element per item in parameter list.
    #generation by type because its conceptually simpler
    if not aspectmod & interact:
        covar = {'k': np.arange(K),
             'a': np.repeat(np.nan, parLength), #why parLength? 
             'type': np.repeat(1, K)}

    if(aspectmod & interact == False):
        covar = {'k': np.append(np.arange(K), np.repeat(np.nan, A)),
                 'a': np.append(np.repeat(np.nan, K), np.arange(A)), 
                 'type': np.append(np.repeat(1, K), np.repeat(2, A))}      
    if(interact):
        covar = {'k': np.append(np.arange(K), np.append(np.repeat(np.nan, A), np.repeat(np.arange(K), A))),
                 'a': np.append(np.repeat(np.nan, K), np.append(np.arange(A), np.repeat(np.arange(A), K))), 
                 'type': np.append(np.repeat(1, K), np.append(np.repeat(2, A),  np.repeat(3,K*A)))}

    kappa = {'out': {'m':m,
                     'params' : np.tile(np.repeat(0,V), (parLength, 1)),
                     'covar' : covar
                     #'kappasum':, why rolling sum?
                    }
            }

    return(kappa['out'])