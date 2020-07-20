import pandas as pd 
import numpy as np
from scipy.optimize import brentq 

def N(z):
    '''
    Normal cumulative density function which returns the 
    cumulative density under the normal curve along the 
    point 'z' where the cumulative density is calculated.
    Refer to scipy.stats documentation for more information
    '''
    from scipy.stats import norm

    return norm.cdf(z)


def call_value(S, K, r, t, vol):
    '''
    Returns the Black-Scholes call option value where
    the parameters have their usual meanings.

    :param S : Underlying stock price
    :param K : Strike price
    :param r : Risk free rate (Usually treasury bond rates or bank rates)
    :param vol : volatility of the stock
    :param t : time to expiration (T - t in documentation)
    '''
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S / K) + t * (r + (vol ** 2.0) / 2))
    d2 = d1 - vol * np.sqrt(t)

    return N(d1) * S - N(d2) * K * np.exp(-r * t)

def put_value(S, K, r, t, vol):
    '''
    Returns the Black-Scholes put option value where
    the parameters have their usual meanings.

    :param S : Underlying stock price
    :param K : Strike price
    :param r : Risk free rate (Usually treasury bond rates or bank rates)
    :param vol : volatility of the stock
    :param t : time to expiration (T - t in documentation)
    '''
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S / K) + t * (r + (vol ** 2.0) / 2))
    d2 = d1 - vol * np.sqrt(t)

    return  N(-d2) * K * np.exp(-r * t) - N(-d1) * S

def phi(x):
    '''
    Phi Helper Function. 
    '''
    import math
    from math import pi
    return np.exp(-0.5 * x**2) / (math.sqrt(2.0 * pi))

def call_delta(S, K, r, t, vol):
    '''
    Black-Scholes Call Delta.
    Partial derivative of the option value 
    with respect to the change in the underlying stock price. 
    Delta measures how the underlying option moves 
    with respect to moves in the underlying stock.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility
    '''
    d1 = (1/(vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol **2.0) * t)
    return N(d1)

def put_delta(S, K, r, t, vol):
    '''
    Black-Scholes Put Delta.
    Partial derivative of the option value 
    with respect to the change in the underlying stock price. 
    Delta measures how the underlying option moves 
    with respect to moves in the underlying stock.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility
    '''
    d1 = (1.0/(vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol ** 2.0) * t)
    
    return N(d1) - 1.0

def gamma(S, K, r, t, vol):
    '''
    Black-Scholes Gamma.
    Second partial derivative of the option value with respect
    to the change in the underlying stock price. Gamma measures movements in delta
    or the convexity in the value of the option with respect to the underlying.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility 
    '''
    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol ** 2.0) * t)

    return phi(d1) / (S * vol * np.sqrt(t))

def vega(S, K, r, t, vol):
    '''
    Black-Scholes Vega. (Returns percentage)
    Partial derivative of the option value with respect to 
    the change in the volatility of the underling. Vega measures 
    how the option price moves with respect to the volatility of the underlying.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility 
    '''
    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol ** 2.0) * t)
    return S * phi(d1) * np.sqrt(t) / 100 

def call_theta(S, K, r, t, vol):
    '''
    Black-Scholes Call Theta(Annualised).
    Partial derivative of the option value with respect to the change in time. 
    Shows the decay of value of option as time passes.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility
    '''

    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol ** 2.0) * t)
    d2 = d1 - (vol * np.sqrt(t)) 

    theta = (-S * phi(d1) * vol) / (2 * np.sqrt(t)) - (r * K * np.exp(-r * t) * N(d2))
    return theta / 365.0

def put_theta(S, K, r, t, vol):
    '''
    Black-Scholes Put Theta(Annualised).
    Partial derivative of the option value with respect to the change in time. 
    Shows the decay of value of option as time passes.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility
    '''

    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol ** 2.0) * t)
    d2 = d1 - (vol * np.sqrt(t)) 

    theta = (-S * phi(d1) * vol) / (2 * np.sqrt(t)) + (r * K * np.exp(-r * t) * N(-d2))
    return theta / 365.0

def call_rho(S, K, r, t, vol):
    '''
    Black-Scholes Call Rho.(Returns Percentage)
    Partial derivative of the option value with respect to change in the risk-free interest rate. 
    Rho measures how the option value changes as the interest rate changes.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility
    '''
    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol ** 2) * t)
    d2 = d1 - (vol * np.sqrt(t))

    rho = K * t * np.exp(-r * t) * N(d2)
    return rho / 100.0

def put_rho(S, K, r, t, vol):
    '''
    Black-Scholes Put Rho.(Returns Percentage)
    Partial derivative of the option value with respect to change in the risk-free interest rate. 
    Rho measures how the option value changes as the interest rate changes.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param vol: volatility
    '''
    d1 = (1.0 / (vol * np.sqrt(t))) * (np.log(S/K) + (r + 0.5 * vol ** 2) * t)
    d2 = d1 - (vol * np.sqrt(t))

    rho = -K * t * np.exp(-r * t) * N(-d2)
    return rho / 100.0

def call_implied_volatility_objective_function(S, K, r, t, vol, call_option_market_price):
    '''
    Objective function which minimises the error between model and market prices.
    This will be sent to the optimizer.

    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param call_option_market rate: market observed option price
    :param vol: volatility of the underlying price
    '''

    return call_option_market_price - call_value(S, K, r, t, vol)

def call_implied_volatility_brent(S, K, r, t, call_option_market_price, a = -2.0, b = 2.0, xtol = 1e-6):
    '''
    Returns the implied volatility using Brent's Algorithm.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param call_option_market_price: market observed option price
    :param a: lower bound for brentq method
    :param b: upper bound for brentq method
    :param xtol: tolerance of error which is considered good enough
    '''

    # avoid mirroring outer scope
    _S, _K, _r, _t, _call_option_market_price = S, K, r, t, call_option_market_price

    # define a nested function that takes our nested param as the input 
    def fcn(vol):

        # returns the difference between market and model price at the given volatility
        return call_implied_volatility_objective_function(_S, _K, _r, _t, vol, _call_option_market_price)

    # first we try to return the results from the brentq algorithm
    try:
        result = brentq(fcn, a = a, b = b, xtol = xtol)

        # if the results are too small, we return nan so that we can later interpolate
        return np.nan if result < xtol else result
    
    # if it fails then we return nan so that we can later interpolate
    except ValueError:
        return np.nan

def call_implied_volatility_objective_function(S, K, r, t, vol, call_option_market_price):
    '''
    Objective function which minimises the error between model and market prices.
    This will be sent to the optimizer.

    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param call_option_market rate: market observed option price
    :param vol: volatility of the underlying price
    '''

    return call_option_market_price - call_value(S, K, r, t, vol)

def call_implied_volatility_brent(S, K, r, t, call_option_market_price, a = -2.0, b = 2.0, xtol = 1e-6):
    '''
    Returns the implied volatility using Brent's Algorithm.
    
    :param S: underlying stock price
    :param K: strike price
    :param r: risk free rate
    :param t: time to expiration
    :param call_option_market_price: market observed option price
    :param a: lower bound for brentq method
    :param b: upper bound for brentq method
    :param xtol: tolerance of error which is considered good enough
    '''

    # avoid mirroring outer scope
    _S, _K, _r, _t, _call_option_market_price = S, K, r, t, call_option_market_price

    # define a nested function that takes our nested param as the input 
    def fcn(vol):

        # returns the difference between market and model price at the given volatility
        return call_implied_volatility_objective_function(_S, _K, _r, _t, vol, _call_option_market_price)

    # first we try to return the results from the brentq algorithm
    try:
        result = brentq(fcn, a = a, b = b, xtol = xtol)

        # if the results are too small, we return nan so that we can later interpolate
        return np.nan if result < xtol else result
    
    # if it fails then we return nan so that we can later interpolate
    except ValueError:
        return np.nan  

def put_implied_volatility_objective_function(S, K, r, t, vol, put_option_market_price):
    '''
    Objective function which minimises the error between model and market prices.
    This will be sent to the optimizer.
    :param S: underlying
    :param K: strike price
    :param r: rate
    :param t: time to expiration
    :param vol: volatility
    :param call_option_market_price: market observed option price
    '''
    return put_option_market_price - put_value(S, K, r, t, vol)

def put_implied_volatility_brent(S, K, r, t, put_option_market_price, a=-2.0, b=2.0, xtol=1e-6):
    '''
    Returns the implied volatility using Brent's Algorithm.
    
    :param S: underlying
    :param K: strike price
    :param r: rate
    :param t: time to expiration
    :param call_option_market_price: market observed option price
    :param a: lower bound for brentq method
    :param b: upper gound for brentq method
    :param xtol: tolerance which is considered good enough
    '''
    
    # avoid mirroring out scope  
    _S, _K, _r, _t, _put_option_market_price = S, K, r, t, put_option_market_price
    
    # define a nsted function that takes our target param as the input
    def fcn(vol):
        
        # returns the difference between market and model price at given volatility
        return put_implied_volatility_objective_function(_S, _K, _r, _t, vol, _put_option_market_price)
    
    # first we try to return the results from the brentq algorithm
    try:
        result = brentq(fcn, a=a, b=b, xtol=xtol)
        
        # if the results are *too* small, sent to np.nan so we can later interpolate
        return np.nan if result <= 1.0e-6 else result
    
    # if it fails then we return np.nan so we can later interpolate the results
    except ValueError:
        return np.nan







 