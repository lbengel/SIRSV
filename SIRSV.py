# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 09:17:22 2021

@author: Lukas Bengel
"""
####################################################################################################
#
# This code is based on the work of R. Elie, E. Hubert and G. Turinici
# "Contact rate epidemic control of COVID-19: an equilibrium view" Math. Model. Nat. Phenom. (2020),
# https://doi.org/10.1051/mmnp/2020022 , which I gratefully acknowledge.
#
####################################################################################################
import numpy as np
from scipy.integrate import odeint
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.optimize import minimize


SMALL_SIZE = 12
MEDIUM_SIZE = 13
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.style.use(['science', 'high-vis'])


# use Science Plots (needs to be installed, see https://pypi.org/project/SciencePlots/)


def cost_effort(x):
    """
    cost of contact reduction
    """
    return (beta_0 / x) - 1


def cost_effort_deriv(x):
    return - beta_0 / x ** 2


def cost_infection(i):
    """
    cost of infection
    """
    return r_I * np.ones_like(i)


#################### vaccination strategies #########################


def vac(time):
    return 0


# def vac(time):
#     """
#     vaccination after 60 days
#     """
#     if time <= 60:
#         delta = 0
#     else:
#         delta = 1/300
#     return delta


########## SIRSV ###########

def deriv_improve(y, time, gamma, beta):
    """
    SIRSV model
    """
    S, logI, R = y

    # SIRSV model
    dSdt = - beta(time) * S * np.exp(logI) + rho * R - vac(time) * S
    dlogIdt = beta(time) * S - gamma
    dRdt = gamma * np.exp(logI) - rho * R
    return dSdt, dlogIdt, dRdt,


def SIRSV_improve(time, beta):
    """
    calculate SIRSV model  
    """
    ret = odeint(deriv_improve, initial_guess_y0, time, args=(gamma, beta))
    S, I, R = ret.T
    return S, np.exp(I), R,


def H(b, u, v, I_t):
    """
    Hamiltonian
    """
    return cost_effort(b) + b * I_t * (v - u)


def HJB_ode(y, time, b, I_func):
    """
    Hamilton-Jacobi-Bellman equation
    """
    u, v, w = y
    dudt = cost_effort(b(T_max - time)) + b(T_max - time) * \
        I_func(T_max - time) * (v - u) - vac(T_max - time) * u
    dvdt = cost_infection(time) + gamma * (w - v)
    dwdt = rho * (u - w)
    return dudt, dvdt, dwdt


def costs(t_interval, b, beta):
    """
    calculate costs
    :param t_interval: time interval
    :param b: contact rate of individual
    :param beta: contact rate of population
    :return: costs
    """
    S, I, R = SIRSV_improve(t_interval, beta)
    I_cont = interpolate.interp1d(
        t_interval, I, kind='cubic', fill_value="extrapolate")

    ret = odeint(HJB_ode, [0, 0, 0], t_interval, args=(b, I_cont))
    u = ret.T[0]
    C = u[-1]
    return C


def fixed_point(t_interval):
    """
    compute Mean Field Nash equilibrium using fixpoint iteration
    """
    b = beta_init
    a = 0.5
    b_ante = beta_init
    cost = np.zeros(number_step)
    b_t = b(t_interval)
    for i in range(number_step):
        cost[i] = costs(t_interval, b, b)
        print(f"Nash step: {i}, Cost: {cost[i]}")
        if (i == number_step - 1):
            b_ante = b

        S, I, R = SIRSV_improve(t_interval, b)
        I_func = interpolate.interp1d(
            t_interval, I, kind='linear', fill_value="extrapolate")
        ret = odeint(HJB_ode, [0, 0, 0], t_interval, args=(b, I_func))

        u = ret.T[0]
        v = ret.T[1]
        u = np.flip(u)
        v = np.flip(v)

        b_t = b(t_interval)
        for i in range(n_max):
            temp = minimize(H, b_t[i], args=(
                u[i], v[i], I[i]), bounds=((0.05, beta_0),))
            temp = temp.x
            b_t[i] = a * temp + (1 - a) * b_t[i]
        b = interpolate.interp1d(
            t_interval, b_t, kind='linear', fill_value="extrapolate")

    res_SIRSV = SIRSV_improve(t_interval, b)

    fig = plt.figure(2, figsize=(10, 4))
    fig.suptitle('Convergence towards the Nash equilibirum', fontsize=16)
    plt.gcf().subplots_adjust(wspace=0.25, hspace=0.25)
    plt.subplot(1, 2, 1)
    plt.plot(range(number_step), cost, "m")
    plt.ylabel('Costs')
    plt.xlabel('Iterations')

    plt.subplot(1, 2, 2)
    plt.plot(t_interval, beta_init(t_interval), "r")
    plt.plot(t_interval, b(t_interval), "m")
    plt.plot(t_interval, b_ante(t_interval), "g")
    plt.legend([r"$\beta^0$", r"$\beta^*$", r"At step $n-1$"],
               fontsize='13', frameon=False)
    plt.xlabel('Days')
    plt.ylabel('Contact rate')

    if save_plots:
        plt.savefig('SIRSV_Convergence_Nash_rI={}_R0={}_i0={}.png'.format(r_I, R_0, I0), dpi=300,
                    bbox_inches='tight')

    return b, b_ante, cost, res_SIRSV


def dynamic_y(y, t_interval, b, S_func, I_func, P_S_func):
    """
    adjoint equation of PMM
    """
    y_S, y_I, y_R, y_P_S, y_P_I, y_P_R = y
    dLdt = [b(T_max - t_interval) * I_func(T_max - t_interval) * (y_I - y_S) - vac(T_max - t_interval) * y_S,
            b(T_max - t_interval) * S_func(T_max - t_interval) * (y_I - y_S) + gamma * (y_R -
            y_I) + (-y_P_S + y_P_I) * b(T_max - t_interval) * P_S_func(T_max - t_interval),
            rho * (y_S - y_R),
            cost_effort(b(T_max - t_interval)) + b(T_max - t_interval) *
            I_func(T_max - t_interval) * (y_P_I - y_P_S)
            - vac(T_max - t_interval) * y_P_S,
            cost_infection(T_max - t_interval) + (y_P_R - y_P_I) * gamma,
            rho * (y_P_S - y_P_R)]
    return dLdt


def deriv_improve_kolmog(y, t, b, I):
    """
    Kolmogorov equation
    """
    P_S, P_I, P_R = y
    dP_Sdt = - P_S * b(t) * I(t) + rho * P_R - vac(t) * P_S
    dP_Idt = P_S * b(t) * I(t) - gamma * P_I
    dP_Rdt = gamma * P_I - rho * P_R
    return dP_Sdt, dP_Idt, dP_Rdt


def improve_p(time, b, I):
    """
    calculate solution of Kolmogorov equation
    """
    ret = odeint(deriv_improve_kolmog, [1, 0, 0], time, args=(b, I))
    P_S, P_I, P_R = ret.T
    return P_S, P_I, P_R


def improve_y(t_interval, b, S_func, I_func, P_S_func):
    """
    calculate solution of adjoint equation
    """
    ret = odeint(dynamic_y, [0, 0, 0, 0, 0, 0],
                 t_interval, args=(b, S_func, I_func, P_S_func))
    y_S = ret.T[0]
    y_I = ret.T[1]
    y_R = ret.T[2]
    y_P_S = ret.T[3]
    y_P_I = ret.T[4]
    y_P_R = ret.T[5]
    return np.flip(y_S), np.flip(y_I), np.flip(y_R), np.flip(y_P_S), np.flip(y_P_I), np.flip(y_P_R)


def gradient(t_interval, beta):
    """
    calculate gradient of Hamiltonian
    """
    S, I, R = SIRSV_improve(t_interval, beta)
    S_cont = interpolate.interp1d(
        t_interval, S, kind='linear', fill_value="extrapolate")
    I_cont = interpolate.interp1d(
        t_interval, I, kind='linear', fill_value="extrapolate")
    P_S, P_I, P_R = improve_p(t_interval, beta, I_cont)
    P_S_func = interpolate.interp1d(
        t_interval, P_S, kind='linear', fill_value="extrapolate")

    y_S, y_I, y_R, y_P_S, y_P_I, y_P_R = improve_y(
        t_interval, beta, S_cont, I_cont, P_S_func)

    grad = cost_effort_deriv(beta(t_interval)) * P_S + \
        S * I * (y_I - y_S) + (y_P_I - y_P_S) * I * P_S
    c = costs(t_interval, beta, beta)
    return grad, c


def cost_anarchy(t_interval):
    """
    calculate societal optium using PMM 
    """
    b = b_nash
    h = 0.001 * np.ones(number_step)
    a = 0.5

    cost = np.zeros(number_step)
    for i in range(number_step):
        grad, cost[i] = gradient(t_interval, b)
        print(f"Societal step: {i}, Cost: {cost[i]}")

        temp = b(t_interval) - h[i] * grad
        temp = np.maximum(np.minimum(
            temp, beta_0 * np.ones_like(temp)), 0.05 * np.ones_like(temp))
        temp = (a * temp + (1 - a) * b(t_interval))

        b = interpolate.interp1d(
            t_interval, temp, kind='linear', fill_value="extrapolate")

    fig = plt.figure(4, figsize=(10, 4))
    fig.suptitle('Convergence towards Societal optimum', fontsize=16)
    plt.gcf().subplots_adjust(wspace=0.25, hspace=0.25)
    plt.subplot(1, 2, 1)
    plt.plot(range(number_step), cost, "m")
    plt.ylabel('Costs')
    plt.xlabel('Iterations')

    plt.subplot(1, 2, 2)
    plt.plot(t_interval, beta_init(t_interval), "r")
    plt.plot(t_interval, b_nash(t_interval), "m")
    plt.plot(t_interval, b(t_interval), "g")
    plt.legend([r"$\beta^0$", r"Nash", r"Soc."], fontsize='13', frameon=False)
    plt.xlabel('Days')
    plt.ylabel('Contact rate')

    if save_plots:
        plt.savefig('SIRSV_Convergence_Societal_rI={}_R0={}_i0={}.png'.format(r_I, R_0, I0), dpi=300,
                    bbox_inches='tight')

    res_SIRSV = SIRSV_improve(t_interval, b)

    return b, h, cost, res_SIRSV


if __name__ == "__main__":
    ################ INITIALISATION ###################

    # A grid of time points (in days)
    T_max = 360
    # n_max = 4*(T_max+1)
    n_max = T_max + 1
    t = np.linspace(0, T_max, n_max)

    # Initial values I0 and R0.
    I0, R0 = 0.01, 0.
    # S0 is the rest
    S0 = 1 - I0 - R0
    logI0 = np.log(I0)

    # Initial conditions vector
    initial_guess_y0 = S0, logI0, R0

    gamma = 1 / 10.0  # time in the latent non infectious phase
    R_0 = 2  # reproduction number
    beta_0 = R_0 * gamma  # transmission rate of the disease in normal conditions
    rho = 0  # rate of lost of immunisation
    r_I = 30  # cost of infection for an individual
    save_plots = False

    R_0_List = [2]
    for R_0 in R_0_List:
        beta_0 = R_0 * gamma

        ############# end initialization #############

        def beta_init(time):
            return beta_0 * np.ones_like(time)

        ################# NO EFFORT ####################

        res_SIRSV_init = SIRSV_improve(t, beta_init)
        S_init, I_init, R_init = res_SIRSV_init

        ################### NASH ##########################

        number_step = 15

        b_nash, b_ante, cost, res_SIRSV_nash = fixed_point(t)
        S_nash, I_nash, R_nash = res_SIRSV_nash

        ################### PLOTTING NASH ##########################

        fig = plt.figure(3, figsize=(10, 10))
        fig.suptitle('Nash Equilibrium', fontsize=16)

        plt.subplot(2, 2, 1)
        plt.plot(t, S_nash, "b", t, I_nash, "r", t, R_nash, "g")
        plt.legend(["$S$", "$I$", "$R$"], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel(r'Proportion of S, I, and R at equilibrium $\beta^\star$')

        plt.subplot(2, 2, 2)
        plt.plot(t, S_init, "b")
        plt.plot(t, S_nash, "b")
        plt.legend([r"$\beta^0$", r"$\beta^*$"], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('Proportion of Susceptible')

        plt.subplot(2, 2, 3)
        plt.plot(t, I_init, "r")
        plt.plot(t, I_nash, "r")
        plt.legend([r"$\beta^0$", r"$\beta^*$"], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('Proportion of Infected')

        plt.subplot(2, 2, 4)
        plt.plot(t, R_init, "g")
        plt.plot(t, R_nash, "g")
        plt.legend([r"$\beta^0$", r"$\beta^*$"], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('Proportion of Recovered')

        if save_plots:
            plt.savefig('SIRSV_Nash_VS_init_rI={}_R0={}_i0={}.png'.format(r_I, R_0, I0), dpi=300,
                        bbox_inches='tight')
        else:
            plt.show()

        ################ OPTIMUM SOCIETAL ########################

        number_step = 50

        b_soc, h_2, cost_2, res_SIRSV_soc = cost_anarchy(t)

        print(cost[-1], cost_2[-1])

        S_soc, I_soc, R_soc = res_SIRSV_soc

        ################### PLOTTING OPT SOC ##########################

        fig = plt.figure(5, figsize=(10, 10))
        fig.suptitle('Comparison Nash and Societal optimum', fontsize=16)

        plt.subplot(2, 2, 1)
        plt.plot(t, b_nash(t), "m")
        plt.plot(t, b_soc(t), "m")
        plt.legend([r"Nash", r"Soc."], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('Contact rate')

        plt.subplot(2, 2, 2)
        plt.plot(t, S_nash, "b")
        plt.plot(t, S_soc, "b")
        plt.legend([r"Nash", r"Soc."], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('$S$')

        plt.subplot(2, 2, 3)
        plt.plot(t, I_nash, "r")
        plt.plot(t, I_soc, "r")
        plt.legend([r"Nash", r"Soc."], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('$I$')

        plt.subplot(2, 2, 4)
        plt.plot(t, R_nash, "g")
        plt.plot(t, R_soc, "g")
        plt.legend([r"Nash", r"Soc."], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('$R$')

        if save_plots:
            plt.savefig('SIRSV_Nash_optimum_rI={}_R0={}_i0={}.png'.format(r_I, R_0, I0), dpi=300,
                        bbox_inches='tight')

        fig = plt.figure(320, figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(t, beta_init(t), "r")
        plt.plot(t, b_nash(t), "m")
        plt.legend(
            [r"$\beta^0$", r"$\beta^{\star}$"], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('Contact rate')

        plt.subplot(1, 2, 2)
        plt.plot(t, I_init, "r")
        plt.plot(t, I_nash, "r")
        plt.legend(
            [r"$\beta^0$", r"$\beta^{\star}$"], fontsize='13', frameon=False)
        plt.xlabel('Days')
        plt.ylabel('$I$')

        if save_plots:
            plt.savefig('SIRSV.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
