#!/usr/bin/env python
"""Quick evaluation of BSLR based on ODE model in Jaeger et al.

Functions:
    main: Generate 3-time and 10-time figures in Frontiers in
        Plant Science manuscript.
    test_causnet: Test CausNet with Jaeger network.
    jaeger_ode: Drift of ODE in Jaeger et al.
    nu_1_0: Basal production of FT.
    reproduce_figure_s2: Reproduce Figure S2 in Jaeger et al.
"""
# Plotting module.
import pandas as pd
from scipy.integrate import odeint
import numpy as np
import sys
if sys.platform == 'darwin':
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
elif sys.platform in ['linux', 'linux2']:
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
else:
    print("No support for Windows.")
    exit(1)

import causnet


def main():
    test_causnet(3)
    test_causnet(10)
    return


param = {'nu_1_35s': 0,
         'nu_2_35s': 0,
         'nu_3_35s': 0,
         'nu_4_35s': 0,
         'nu_5_35s': 0,
         'eta_leaf': 0.01,
         'max_rate': 0.11,
         'nu_2_0': 0.01,
         'nu_3_0': 0.01,
         'nu_4_0': 0.01,
         'nu_5_0': 0,
         'nu_2_+': 0.05,
         'nu_3_+': 0.05,
         'nu_4_+': 0.05,
         'nu_5_+': 0.05,
         'nu_2_++': 0.1,
         'nu_3_++': 0.1,
         'nu_4_++': 0.1,
         'nu_5_++': 0.1,
         'K_13': 0.39381,
         'K_23': 3.2556,
         'K_4_3': 0.28203,
         'h_4_3': 4.00,
         'K_23_4': 9.3767,
         'h_23_4': 3.8497,
         'K_13_4': 0.040555,
         'h_13_4': 4.00,
         'K_23_5': 0.033666,
         'h_23_5': 4.00,
         'K_13_5': 0.029081,
         'h_13_5': 1.8217,
         'K_4_5': 0.13032,
         'h_4_5': 3.9369,
         'K_5_4': 0.28606,
         'h_5_4': 3.6732,
         'h_5_2': 1.0239,
         'K_5_2': 0.2,
         'delta_1': 0.1,
         'delta_2': 0.1,
         'delta_3': 0.1,
         'delta_4': 0.1,
         'delta_5': 0.1}


def test_causnet(num_times, rand_seed=0, sig_level=0.05, output=''):
    """Test CausNet using network in Jaeger et al.

    Args:
        num_times: int
            Number of times.
        rand_seed: int
            Random number generator seed.
        sig_level: float
            Significance level.
        output: str
            Output filename.

    Returns:
        Saves graph file.
    """
    np.random.seed(rand_seed)
    x_init = np.random.rand(10, 5)
    times = np.linspace(0, 50, num_times)
    x = np.empty((10*num_times, 5))
    for i in range(10):
        x[i*num_times:(i+1)*num_times, :] = odeint(
            jaeger_ode, x_init[i, :], times, args=(param,)
            )
    df = pd.DataFrame(data=x.T, columns=np.arange(num_times*10),
                      index=['G1', 'G2', 'G3', 'G4', 'G5'])
    df.to_csv('exp-test-t{}-c10.csv'.format(num_times))
    if not output:
        output = (
            'test-t{num_times}-c10-bslr-s{sig_level}'
            '-r{rand_seed}.xml'.format(
                num_times=num_times, sig_level=sig_level,
                rand_seed=rand_seed
                )
            )
    causnet.main(
        '-c cond-jaeger-c10-t{num_times}.txt -p 1 '
        '-i gene-list-jaeger.csv -g {output} '
        '-x exp-test-t{num_times}-c10.csv '
        '-P parser-test-t{num_times}-c10.csv '
        '-f {sig_level}'.format(
            output=output, num_times=num_times,
            sig_level=sig_level
            ).split()
        )
    return


def jaeger_ode(x, t, p):
    """Drift of the ODE in Jaeger et al.

    Args:
        x: array
            Gene expression levels (mRNA abundances).
            x[0]: FT.
            x[1]: TFL1.
            x[2]: FD.
            x[3]: LFY.
            x[4]: AP1.
        t: float
            Time.
        p: dict
            Parameters.
    """
    x13 = x[0]/p['K_13']/(1+x[0]/p['K_13']+x[1]/p['K_23'])*x[2]
    x23 = x[1]/p['K_23']/(1+x[0]/p['K_13']+x[1]/p['K_23'])*x[2]
    return np.asarray([
        p['nu_1_35s']+nu_1_0(
            t, p['eta_leaf'], p['max_rate']
            )-p['delta_1']*x[0],
        (p['nu_2_35s']+((x[4]/p['K_5_2'])**p['h_5_2'])
         /(1+(x[4]/p['K_5_2'])**p['h_5_2'])*p['nu_2_0']
         + 1/(1+(x[4]/p['K_5_2'])**p['h_5_2'])*p['nu_2_+']
         - p['delta_2']*x[1]),
        (p['nu_3_35s']+1/(1+(x[3]/p['K_4_3'])**p['h_4_3'])
         *p['nu_3_0']
         +((x[3]/p['K_4_3'])**p['h_4_3'])
         /(1+(x[3]/p['K_4_3'])**p['h_4_3'])*p['nu_3_+']
         - p['delta_3']*x[2]),
        (p['nu_4_35s']+1/(
            1+(x13/p['K_13_4'])**p['h_13_4']
            +(x23/p['K_23_4'])**p['h_23_4']
            )/(1+(x[4]/p['K_5_4'])**p['h_5_4'])*p['nu_4_0']
         +((x13/p['K_13_4'])**p['h_13_4']
           +(x[4]/p['K_5_4'])**p['h_5_4'])
         /(1+(x13/p['K_13_4'])**p['h_13_4']
           +(x23/p['K_23_4'])**p['h_23_4'])
         /(1+(x[4]/p['K_5_4'])**p['h_5_4'])*p['nu_4_+']
         +((x13/p['K_13_4'])**p['h_13_4']
           *(x[4]/p['K_5_4'])**p['h_5_4'])
         /(1+(x13/p['K_13_4'])**p['h_13_4']
           +(x23/p['K_23_4'])**p['h_23_4'])
         /(1+(x[4]/p['K_5_4'])**p['h_5_4'])*p['nu_4_++']
         - p['delta_4']*x[3]),
        (p['nu_5_35s']+1/(
            1+(x13/p['K_13_5'])**p['h_13_5']
            +(x23/p['K_23_5'])**p['h_23_5']
            )/(1+(x[3]/p['K_4_5'])**p['h_4_5'])*p['nu_5_0']
         +((x13/p['K_13_5'])**p['h_13_5']
           +(x[3]/p['K_4_5'])**p['h_4_5'])
         /(1+(x13/p['K_13_5'])**p['h_13_5']
           +(x23/p['K_23_5'])**p['h_23_5'])
         /(1+(x[3]/p['K_4_5'])**p['h_4_5'])*p['nu_5_+']
         +((x13/p['K_13_5'])**p['h_13_5']
           *(x[3]/p['K_4_5'])**p['h_4_5'])
         /(1+(x13/p['K_13_5'])**p['h_13_5']
           +(x23/p['K_23_5'])**p['h_23_5'])
         /(1+(x[3]/p['K_4_5'])**p['h_4_5'])*p['nu_5_++']
         - p['delta_5']*x[4]),
        ])


def nu_1_0(t, eta_leaf, max_rate):
    """Basal production with nothing bound, excluding 
    the 35S promoter effect.

    Args:
        t: float
            Time.
        eta_leaf: float
            Constant rate with time.
        max_rate: float
            Maximum basal production rate.

    Return: float
        Basal production rate.
    """
    return min(eta_leaf*t, max_rate)


def reproduce_figure_s2():
    x_init = [0, 0.6, 0.1, 0.1, 0]
    times = np.linspace(0, 50, 101)
    x = odeint(jaeger_ode, x_init, times, args=(param,))
    fig, ax = plt.subplots()
    ax.plot(times, x[:, 0], color='red')
    ax.plot(times, x[:, 1], color='blue')
    ax.plot(times, x[:, 2], color='grey')
    ax.plot(times, x[:, 3], color='green')
    ax.plot(times, x[:, 4], color='yellow')
    plt.legend(['FT', 'TFL1', 'FD', 'LFY', 'AP1'], loc='best')
    fig.savefig('reproduced-figure-s2.eps')


if __name__ == "__main__":
    main()
