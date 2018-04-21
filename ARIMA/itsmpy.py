"""
Repackage package itsmr in R
https://cran.r-project.org/web/packages/itsmr/itsmr.pdf
"""
import sys
import numpy as np
import itertools
import math
import warnings
import pprint
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acovf

class ITSM:
    def aacvf(self, a, h):
        """
        Computes autocovariance vector for an ARMA model
        :type a: dictionary, dictionary of phi, theta, sigma2
        :type h: int, maximum lag
        :rtype gamma: np.array, to accomodate lag 0 at index 0
        """
        phi = a['phi']
        theta = a['theta']
        sigma2 = a['sigma2']
        p = len(phi)
        q = len(theta)
        psi = self.ma_inf(a,q)
        theta = np.append(1, theta)
        def f(k):
            return np.sum(theta[(k-1):(q+1)]*psi[0:(q-k+2)])
        r = np.zeros((max(p+1,q+1,h+1),))
        r[0:(q+1)] = sigma2*np.array(list(map(f, np.arange(1,(q+2)))))
        #solve for gamma in A*gamma=r
        def f2(k):
            return np.concatenate((np.zeros(p-k),np.append(1,-phi), np.zeros(k)))
        A = np.array(list(map(f2, np.arange(0,p+1)))).reshape((-1,int(2*p+1)))
        A = np.hstack((A[:,p:(p+1)], A[:,(p+1):(2*p+1)]+A[:,np.arange(0,p)[::-1]]))
        gamma = np.zeros(max(p+1,h+1))
        gamma[0:(p+1)] = np.linalg.lstsq(A,r[0:(p+1)], rcond=None)[0]
        #calculate remaining lags recursively
        if h>p:
            for k in range(p+1,h+1):
                gamma[k] = r[k] + np.sum(phi*gamma[k-np.arange(0,p)-1])
        elif h<p:
            gamma = gamma[0:(h+1)]
        return gamma

    def trend(self, x, p):
        """
        Calculates the trend component of data
        :type x: np.array, time series data
        :type p: int, polynomial order (1=linear, 2=quadratic)
        :rtype xhat: np.array, trend component
        """
        n = len(x)
        X = np.array([])
        for i in range(p+1):
            s = np.arange(1,(n+1)) ** i
            X = np.hstack((X, s))
        X = X.reshape((n,(p+1)), order = 'F')
        qr = np.linalg.qr(X, mode='economic')
        b = np.linalg.lstsq(X,x)[0]
        xhat = np.dot(X,b)
        return xhat

    def smooth_ma(self, x, q):
        """
        Smooth data with a moving average filter
        :type x: np.array, time series data
        :type q: int, window size=2q+1
        :rtype m: np.array, x with MA filter
        """
        n = len(x)
        x = np.concatenate((np.repeat(x[0],int(q)),x,np.repeat(x[n-1],int(q))))
        qq = np.arange(int(-q), int(q+1))
        def F(t):
            return x[t+qq-1].sum() / (2*q+1)
        m = np.array(list(map(F, np.arange(int(q+1),int(n+q+1)))))
        return m

    def season(self, x, d):
        """
        Calculates the seasonal component of data
        :type x: np.array, time series data
        :type d: int, number of observation per season
        :rtype s: np.array, seasonal components
        """
        n = len(x)
        q = np.floor(d/2)
        if d==2*q:
            def F1(t):
                return x[int(t-q-1)]/2 + sum(x[int(t-q):int(t+q-1)]) + x[int(t+q-1)]/2
            m = np.array(list(map(F1, np.arange(q+1,n-q+1)))) / d
            m = np.concatenate((np.repeat(0,q),m,np.repeat(0,q)))
        else:
            m = self.smooth_ma(x,q)
        dx = x - m
        def F2(k):
            return dx[np.arange(int(k+q-1), int(n-q), d)].mean()
        w = np.array(list(map(F2, np.arange(1,d+1))))
        w = w - w.mean()
        s = np.tile(w,int(n+q))[0:int(n+q)][int(q):int(q+n)]
        return s

    def ma_inf(self, a, n=50):
        """
        Moving Average model with infinite order
        :type a: dictionary, ARMA model coefficients phi and theta
        :type n: int, required order of psi
        :rtype psi_new: np.array,
            coefficient vector of length n+1 to accomodate psi_0 at index 0
        """
        if n==0:
            return 1
        theta = np.append(a['theta'], np.zeros(n))
        phi = a['phi']
        p = len(phi)
        psi = np.append(np.append(np.zeros((p,)),1), np.zeros((n,)))
        for j in range(n):
            psi[j+p+1] = theta[j] + np.sum(phi*psi[np.arange(0,p)[::-1]+j+1])
        psi_new = psi[np.arange(0,n+1)+p]
        return psi_new

    def arma(self, x, p=0, q=0):
        """
        Estimates ARMA model parameters using maximum likelihood
        :type x: np.array, time series data
        :type p: int, AR order
        :type q: int, MA order
        :rtype a: dictionary,
            - phi: AR coefficients
            - theta: MA coefficients
            - sigma2: white noise variance
            - aicc: Akaike information criterion corrected
            - se_phi: standard errors for phi
            - se_theta: standard errors for theta
        """
        x = x - x.mean()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
        try:
            model_fit = ARIMA(x, (p,0,q)).fit(disp=False)
        except: pass
        c = np.append(model_fit.params[1:],model_fit.params[0])
        v = np.diag(model_fit.cov_params())
        v = np.append(v[1:], v[0])
        if p==0:
            phi = 0
            se_phi = 0
        else:
            phi = c[0:p]
            se_phi = np.sqrt(v[0:p])
        if q==0:
            theta = 0
            se_theta = 0
        else:
            theta = c[p:(p+q)]
            se_theta = np.sqrt(v[p:(p+q)])
        a = {'phi': phi, 'theta': theta, 'sigma2': None,
             'aicc': None, 'se_phi': se_phi, 'se_theta': se_theta}
        a = self.innovation_update(x, a)
        return a

    def autofit(self, x, p=6, q=6):
        """
        Fits the best ARMA model
        :type x: np.array, time series data
        :type p: int, AR highest order
        :type q: int, MA highest order
        :rtype a: dictionary
            - phi: AR coefficients
            - theta: MA coefficients
            - sigma2: white noise variance
            - aicc: Akaike information criterion corrected
            - se_phi: standard errors for phi
            - se_theta: standard errors for theta
        """
        a = {'aicc': float('Inf')}
        pp = range(6)
        qq = range(6)
        pq = list(itertools.product(pp,qq))[1:]
        for i in pq:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
            try:
                b = self.arma(x, i[0], i[1])
                if b['aicc'] < a['aicc']: a = b
            except: pass
        return a

    def forecast(self, x, M, a, h=10, opt=2, alpha=0.5):
        """
        Forecasts future values of a time series
        :type x: np.array, time series data
        :type M: list, data model ([] for None)
        :type a: dictionary, ARMA model coefficients (phi, theta, sigma2)
        :type h: int, number of predicted values
        :type opt: display option
        :type alpha: float, significance level (default=0.05)
        :rtype f: dictionary,
            - pred: predicted values
            - se: standard errors (not included if log function in M)
            - l: lower 95% prediction bound
            - u: upper 95% prediction bound
        """
        f = self.forecast_transform(x, M, a, h, 1)
        # compute standard erros
        if 'phi' in f:
            psi = self.ma_inf({'phi': f['phi'], 'theta': a['theta']}, h)
            def g(j):
                return np.sum(psi[:j]**2)
            var = np.array(list(map(g, np.arange(1,h+1))))
            se = np.sqrt(a['sigma2']*var)
            q = norm.ppf(1-alpha/2)*se
            l = f['pred'] - q
            u = f['pred'] + q
            f = {'pred': f['pred'], 'se': se, 'l': l, 'u': u}
        return f

    def forecast_transform(self, x, M, a, h, k):
        """
        Transforms the data, forecasts, inverts the transform
        """
        if k>len(M):
            return self.forecast_arma(x,a,h)
        if M[k-1] == "log":
            return self.forecast_log(x,M,a,h,k)
        if M[k-1] == "season":
            return self.forecast_season(x,M,a,h,k)
        if M[k-1] == "trend":
            return self.forecast_trend(x,M,a,h,k)
        else:
            sys.exit('No algorithm specified!')

    def forecast_arma(self, x, a, h):
        n = len(x)
        mu = x.mean()
        x = x - mu
        dx = self.innovations(x, a)
        dx = np.append(dx, np.zeros(h))
        x = np.append(x, np.zeros(h))

        phi = a['phi']
        theta = a['theta']
        p = len(phi)
        q = len(theta)
        # forecast h steps ahead
        for t in range(n, n+h):
            A = np.sum(phi * x[np.arange(t-p, t)[::-1]])
            B = np.sum(theta * dx[np.arange(t-q, t)[::-1]])
            x[t] = A+B
        pred = x[n:(n+h)] + mu
        results = {'pred': pred, 'phi': phi}
        return results

    def forecast_log(self, x, M, a, h, k):
        f = self.forecast_transform(np.log(x), M, a, h, k+1)
        # prediction bounds
        if 'phi' not in f:
            l = f['l']
            u = f['u']
        else:
            psi = self.ma_inf({'phi': f['phi'], 'theta': a['theta']}, h)
            def g(j):
                return np.sum(psi[:j]**2)
            var = np.array(list(map(g, np.arange(1,h+1))))
            se = np.sqrt(a['sigma2']*var)
            l = f['pred'] - 1.96*se
            u = f['pred'] + 1.96*se
        pred = np.exp(f['pred'])
        l = np.exp(l)
        u = np.exp(u)
        d = {'pred': pred, 'l': l, 'u': u}
        return d

    def forecast_season(self, x, M, a, h, k):
        n = len(x)
        # d: number of observaions per season
        d = M[k]
        # m: estimated trend
        q = np.floor(d/2)
        if d==2*q:
            def F1(t):
                return x[int(t-q-1)]/2 + sum(x[int(t-q):int(t+q-1)]) + x[int(t+q-1)]/2
            m = np.array(list(map(F1, np.arange(q+1,n-q+1)))) / d
            m = np.concatenate((np.repeat(0,q),m,np.repeat(0,q)))
        else:
            m = self.smooth_ma(x,q)
        # w: average deviation
        dx = x - m
        def F2(k):
            return dx[np.arange(int(k+q-1), int(n-q), d)].mean()
        w = np.array(list(map(F2, np.arange(1,d+1))))
        w = w - w.mean()
        # s: seasonal component
        s = np.tile(w,int(n+q+h))[int(q):int(q+n+h)]
        # subtract the seasonal component
        y = x - s[:n]
        # forecast
        f = self.forecast_transform(y, M, a, h, k+2)
        # restore the seasonal component
        f['pred'] = f['pred'] + s[n:(n+h)]
        return f

    def forecast_trend(self, x, M, a, h, k):
        n = len(x)
        # p: order of the trend (1=linear, 2=quadratic)
        p = M[k]
        # X: design matrix
        X = np.array([])
        for i in range(p+1):
            s = np.arange(1,(n+h+1)) ** i
            X = np.hstack((X, s))
        X = X.reshape((n+h,(p+1)), order = 'F')
        # b: regression coefficient vector
        b = np.linalg.lstsq(X[:n,:],x)[0]
        # xhat: trend component
        xhat = np.dot(X,b)
        # subtract trend component
        y = x - xhat[:n]
        # forecast
        f = self.forecast_transform(y,M, a, h, k+2)
        # restore trend component
        f['pred'] = f['pred'] + xhat[n:(n+h)]
        return f

    def Resid(self, x, M=[], a=[]):
        """
        Generates model residuals
        :type x: np.array, time series data
        :type M: list, data model ([] for None)
        :type a: dictionary, ARMA model coefficients (phi, theta, sigma2)
        :rtype y: np.array, residuals
        """
        y = x
        k = 1
        while k < len(M):
            if M[k-1] == 'diff':
                lag = M[k]
                y = diff(y, lag, 1) #bug
                k = k+2
            elif M[k-1] == 'log':
                y = np.log(y)
                k = k+1
            elif M[k-1] == 'season':
                d = int(M[k])
                y = y - self.season(y, d)
                k = k+2
            elif M[k-1] == 'trend':
                p = int(M[k])
                y = y - self.trend(y, p)
                k = k+2
            else: break
        y = y - y.mean()
        if len(a)!=0:
            I = self.innovation_kernel(y,a)
            y = (y - I['xhat']) / np.sqrt(I['v'])
        return y

    def innovations(self, x, a):
        """
        innovations for an ARMA model
        :type x: np.array, time series data
        :type a: dictionary, ARMA model coefficients (phi, theta, sigma2)
        :rtype x-xhat: np.array
        """
        xhat = self.innovation_kernel(x,a)['xhat']
        return x-xhat

    def innovation_kernel(self, x, a):
        """
        Calculates xhat and v per ITSF
        :type x: np.array, time series data
        :type a: dictionary, ARMA model coefficients (phi, theta, sigma2)
        :rtype dd: dictionary,
            - xhat: np.array, estimate of data
            - v: mean square error
        """
        #compute autocovariance kappa(i,j)
        #optimized for i>=j and j>0
        def kappa(i,j):
            if j>m:
                return np.sum(theta_r[:(q+1)]*theta_r[(i-j):(i-j+q+1)])
            elif i>(2*m):
                return 0
            elif i>m:
                return (gamma[i-j] - np.sum(phi*gamma[np.abs(np.arange(1-i+j, p-i+j+1))]))/sigma2
            else: return gamma[i-j]/sigma2
        phi = a['phi']
        theta = a['theta']
        sigma2 = a['sigma2']
        N = len(x)
        theta_r = np.append(np.append(1, theta), np.zeros(N))
        # Autocovariance of the model
        gamma = self.aacvf(a, N-1)
        # Innovations algorithm
        p = len(phi)
        q = len(theta)
        m = max(p,q)
        Theta = np.zeros((N-1, N-1))
        v = np.zeros(N)
        v[0] = kappa(1,1)

        for n in range(1,N):
            for k in range(n):
                u = kappa(n+1,k+1)
                s = 0
                if k>0:
                    s = np.sum(Theta[(k-1),np.arange(k)[::-1]]*Theta[(n-1),np.arange(n-k,n)[::-1]]*v[:k])
                Theta[(n-1),(n-k-1)] = (u-s) / v[k]
            s = np.sum(((Theta[(n-1),np.arange(n)[::-1]])**2) * v[:n])
            v[n] = kappa(n+1,n+1) - s
        # compute xhat per equation
        xhat = np.zeros(N)
        if m>1:
            for n in range(1,m):
                xhat[n] = np.sum(Theta[(n-1),:n]*(x[np.arange(n)[::-1]]-xhat[np.arange(n)[::-1]]))
        for n in range(m,N):
            A = np.sum(phi*x[np.arange(n-p,n)[::-1]])
            B = np.sum(Theta[(n-1),:q]*(x[np.arange(n-q,n)[::-1]]-xhat[np.arange(n-q,n)[::-1]]))
            xhat[n] = A + B
        dd = {'xhat': xhat, 'v': v}
        return dd

    def innovation_update(self, x, a):
        """
        Updates AICC and WN variance of an ARMA model
        :type x: np.array, time series data
        :type a: dictionary, ARMA model coefficients (phi, theta, sigma2)
        :rtype a: dictionary, updated ARMA model coefficients (phi, theta, sigma2)
        """
        a['sigma2'] = 1
        I = self.innovation_kernel(x,a)
        xhat = I['xhat']
        v = I['v']

        n = len(x)
        sigma2 = sum((x-xhat)**2/v)/n
        if(np.sum(a['phi']==0)==len(a['phi'])):
            p = 0
        else:
            p = len(a['phi'])
        if(np.sum(a['theta']==0)==len(a['theta'])):
            q = 0
        else:
            q = len(a['theta'])
        loglike = -(n/2)*np.log(2*math.pi*sigma2) - np.sum(np.log(v))/2 - n/2
        aicc = -2*loglike + 2*(p+q+1)*n/(n-p-q-2)
        a['sigma2'] = sigma2
        a['aicc'] = aicc
        return a
