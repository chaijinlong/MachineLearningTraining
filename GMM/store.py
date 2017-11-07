
    def partialEM(self, params):
        k = len(params['mu'])
        l = self.loss(params)

        # lList = [l[0,0]] ##########################
        it = 0
        while it < self.iterLimit:
        # while True:
            it = it + 1
            oldL = l
            gamma = np.zeros((k, self.m))

            # 计算后验概率
            for i in range(k):
                for j in range(self.m):
                    gamma[i, j] = self.getGamma(self.D[:, j].reshape(self.n, 1), i, params)

            # 更新参数
            sumGamma = np.sum(gamma, axis=1, keepdims=True) + 1e-100

            # plt.figure(self.age)####################
            # plt.plot(self.D[0,:], self.D[1,:], 'b+') ####################
            # plt.ion()####################

            for i in range(k):
                newMu = np.sum(self.D * gamma[i, :], axis=1, keepdims=True) / sumGamma[i, 0]
                params['mu'][i] = newMu
                params['s'][i] = np.dot(gamma[i, :] * (self.D - newMu), (self.D - newMu).T) / sumGamma[i, 0]
                params['alpha'][i] = sumGamma[i, 0] / self.m
                # plt.plot(newMu[0,0], newMu[1,0], 'r*') #################

            l = self.loss(params)
            # lList.append(l[0,0]) ######################

            # if l - oldL < -1:
            #     return oldL

            if np.abs(l / oldL - 1) < 10e-6:
            # if np.abs(l / oldL - 1) < 1e-1:
                return l
        return l

    def getGamma(self, x, k, params):
        pM = params['alpha'][k] * self.gaussian(x, params['mu'][k], params['s'][k])
        pMSum = 1e-100  # 1e-100防止pMSum为0
        for l in range(len(params['mu'])):
            pMSum = pMSum + params['alpha'][l] * self.gaussian(x, params['mu'][l], params['s'][l])
        pM = pM / pMSum

        return pM