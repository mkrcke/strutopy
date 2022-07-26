def log_perplexity(self, chunk, total_docs=None):
        """Calculate and return per-word likelihood bound, using a chunk of documents as evaluation corpus.
        Also output the calculated statistics, including the perplexity=2^(-bound), to log at INFO level.
        Parameters
        ----------
        chunk : list of list of (int, float)
            The corpus chunk on which the inference step will be performed.
        total_docs : int, optional
            Number of docs used for evaluation of the perplexity.
        Returns
        -------
        numpy.ndarray
            The variational bound score calculated for each word.
        """
        if total_docs is None:
            total_docs = len(chunk)
        corpus_words = sum(cnt for document in chunk for _, cnt in document)
        subsample_ratio = 1.0 * total_docs / len(chunk)
        perwordbound = self.bound(chunk, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        logger.info(
            "%.3f per-word bound, %.1f perplexity estimate based on a held-out corpus of %i documents with %i words",
            perwordbound, np.exp2(-perwordbound), len(chunk), corpus_words
        )
        return perwordbound

        
def bound(self, corpus, gamma=None, subsample_ratio=1.0):
        """Estimate the variational bound of documents from the corpus as E_q[log p(corpus)] - E_q[log q(corpus)].
        Parameters
        ----------
        corpus : iterable of list of (int, float), optional
            Stream of document vectors or sparse matrix of shape (`num_documents`, `num_terms`) used to estimate the
            variational bounds.
        gamma : numpy.ndarray, optional --> eta for the STM 
            Topic weight variational parameters for each document. If not supplied, it will be inferred from the model.
        subsample_ratio : float, optional
            Percentage of the whole corpus represented by the passed `corpus` argument (in case this was a sample).
            Set to 1.0 if the whole corpus was passed.This is used as a multiplicative factor to scale the likelihood
            appropriately.
        Returns
        -------
        numpy.ndarray
            The variational bound score calculated for each document.
        """
        score = 0.0
        _lambda = self.state.get_lambda()
        Elogbeta = dirichlet_expectation(_lambda)

        for d, doc in enumerate(corpus):  # stream the input doc-by-doc, in case it's too large to fit in RAM
            if d % self.chunksize == 0:
                logger.debug("bound: at document #%i", d)
            if gamma is None:
                gammad, _ = self.inference([doc])
            else:
                gammad = gamma[d]
            Elogthetad = dirichlet_expectation(gammad)

            assert gammad.dtype == self.dtype
            assert Elogthetad.dtype == self.dtype

            # E[log p(doc | theta, beta)]
            score += sum(cnt * logsumexp(Elogthetad + Elogbeta[:, int(id)]) for id, cnt in doc)

            # E[log p(theta | alpha) - log q(theta | gamma)]; assumes alpha is a vector
            score += np.sum((self.alpha - gammad) * Elogthetad)
            score += np.sum(gammaln(gammad) - gammaln(self.alpha))
            score += gammaln(np.sum(self.alpha)) - gammaln(np.sum(gammad))

        # Compensate likelihood for when `corpus` above is only a sample of the whole corpus. This ensures
        # that the likelihood is always roughly on the same scale.
        score *= subsample_ratio

        # E[log p(beta | eta) - log q (beta | lambda)]; assumes eta is a scalar
        score += np.sum((self.eta - _lambda) * Elogbeta)
        score += np.sum(gammaln(_lambda) - gammaln(self.eta))

        if np.ndim(self.eta) == 0:
            sum_eta = self.eta * self.num_terms
        else:
            sum_eta = np.sum(self.eta)

        score += np.sum(gammaln(sum_eta) - gammaln(np.sum(_lambda, 1)))

        return score