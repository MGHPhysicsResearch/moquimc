import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm import *
import pandas as pd
from scipy.optimize import *


def calculate_density(hu):
    if hu < -1000:
        hu = -1000
    if hu > 2995:
        hu = 2995
    density_correction = 1.0
    if hu < -1000:
        density = (hu + 1000) * 0.001029700665188 + 0.00121
    elif hu < -98:
        density = (hu + 0) * 0.000893 + 1.018

    elif hu < 15:
        density = (hu + 1000) * 0.0 + 1.03

    elif hu < 23:
        density = (hu + 0) * 0.001169 + 1.003
    elif hu < 101:
        density = (hu + 0) * 0.000592 + 1.017
    elif hu < 2001:
        density = (hu - 2000) * 0.0005 + 2.201
    elif hu < 2995:
        density = (hu + 0) * 0.0 + 4.54

    density *= density_correction
    return density


def eq0_rad_len(rho, a, b):
    return a + b * rho


def eq0_rsp(x, a, b, c, d, e, f, g, h):
    energy = x
    # return a + b * (c + energy**d) + e * energy
    # return a + b * np.log(energy + c)
    # return a + b * energy + c * energy**2
    # return a + b * np.exp(energy + c)
    # return a + b * energy #+ c * energy**2# + d * energy**3
    return a + b * np.log(energy + c) + d * np.log(energy ** 2 + e)


def fit_rad_len(topas_data, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    topas_mat_name = topas_data['Material Name']
    density_topas = topas_data['Density']
    rad_len_topas = topas_data['Radiation Length']
    density_water = 1.0
    rad_len_water = 36.083
    hus = np.arange(-1024, 2001)
    density_topas_ct = np.zeros(len(hus), dtype=np.float32)
    rad_len_topas_ct = np.zeros(len(hus), dtype=np.float32)

    for i in range(len(topas_mat_name)):
        if 'PatientTissueFromHU' in topas_mat_name[i]:
            mat_name = topas_mat_name[i]
            mat_name = mat_name.replace('PatientTissueFromHU', '')
            if 'Negative' in mat_name:
                mat_name = mat_name.replace('Negative', '')
                hu_value = -1 * int(mat_name)
            else:
                hu_value = int(mat_name)
            # if hu_value == -1024:
            #     continue
            assert hu_value >= -1024 and hu_value <= 2000
            density_topas_ct[hu_value + 1024] = density_topas[i]
            rad_len_topas_ct[hu_value + 1024] = rad_len_topas[i]

    density_partitian = np.array([
        0.095, 0.902, 0.935, 0.96, 0.984, 1.0, 1.0175, 1.08, 1.1035, 1.15, 1.21,
        1.28, 1.34, 1.41, 1.48, 1.54, 1.61, 1.68, 1.74, 1.81, 1.88, 1.95, 2.01
    ])

    print(density_partitian)
    equation_list = [eq0_rad_len] * (len(density_partitian) + 1)
    outliers = [np.inf] * (len(density_partitian) + 1)
    # outliers[2:4]=[0.986,-0.98]
    outliers[0:3] = [-0.987, 0.987, -0.96]
    outliers[8] = -0.978
    popt_total, pcov_total = fit_rad_len_partitian(density_topas_ct,
                                                   rad_len_topas_ct,
                                                   equation_list,
                                                   density_partitian, outliers,
                                                   save_path)
    density_results = np.arange(np.min(density_topas_ct),
                                np.max(density_topas_ct), 0.001)
    # density_results = np.copy(density_topas_ct)
    rad_len_results = np.zeros(len(density_results), dtype=np.float32)
    for i in range(len(density_results)):
        if density_results[i] <= density_partitian[0]:
            ind = 0
        else:
            ind = np.max(
                np.where(density_partitian < density_results[i])[0]) + 1
        equation = equation_list[ind]
        coefficients = popt_total[ind]
        rad_len_results[i] = equation(density_results[i], *coefficients)
    rad_len_plot = density_water * rad_len_water / (rad_len_results *
                                                    density_results)
    f_plot = density_water * rad_len_water / (rad_len_topas_ct *
                                              density_topas_ct)

    discontinue_ind = np.array(
        np.where(
            np.abs(rad_len_results[1:] - rad_len_results[:-1]) > 0.02)[0]) + 1
    density_results2 = np.copy(density_results)
    density_results2 = np.insert(density_results2, list(discontinue_ind),
                                 np.nan)
    rad_len_results = np.insert(rad_len_results, list(discontinue_ind), np.nan)

    density_plot1 = np.copy(density_results)
    density_plot1 = np.insert(density_plot1, list(discontinue_ind), np.nan)
    rad_len_plot = np.insert(rad_len_plot, list(discontinue_ind), np.nan)

    plt.figure(1)
    plt.clf()
    plt.plot(density_topas_ct[::40],
             rad_len_topas_ct[::40],
             'k+',
             label='TOPAS')
    plt.plot(density_plot1, rad_len_plot, 'r--', label='Fitted')
    plt.title('Fitting for density to radiation length conversion')
    plt.xlabel('Density (g/cm3)')
    plt.ylabel('Radiation length (cm)')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(save_path + '/rad_len_fit_total.png')

    plt.figure(1)
    plt.clf()
    plt.plot(density_topas_ct[::40], f_plot[::40], 'k+', label='TOPAS')
    plt.plot(density_results2, rad_len_results, 'r--', label='Fitted')
    plt.title('Fitting for density to radiation length conversion')
    plt.xlabel('Density (g/cm3)')
    plt.ylabel('Radiation length')
    plt.legend(loc='best')
    plt.savefig(save_path + '/rad_len_f_fit_total.png')

    f = open(save_path + '/rad_len_fit.txt', 'w')
    for i in range(len(equation_list)):
        if i == 0:
            # print('if (density <= {:.4f}) {{ {:.4f} + {:.4e} * density }}'.
            #       format(density_partitian[i], *popt_total[i]))
            f.write(
                'if (density <= {:.5f}) {{ f ={:.4f} + {:.4e} * density; }}'.
                format(density_partitian[i], *popt_total[i]))
        elif i == len(equation_list) - 1:
            # print('else if( density>{:.4f}) {{ {:.4f} + {:.4e} * density}}'.
            #       format(density_partitian[i - 1], *popt_total[i]))
            f.write(
                'else if( density>{:.5f}) {{f = {:.4f} + {:.4e} * density;}}'.
                format(density_partitian[i - 1], *popt_total[i]))
        else:
            # print('else if({:.4f} < density <= {:.4f}){{ {:.4f} + {:.4e} * density}}'.
            #       format(density_partitian[i - 1], density_partitian[i],
            #              *popt_total[i]))
            f.write(
                'else if({:.5f} < density <= {:.4f}){{ f ={:.4f} + {:.4e} * density;}}'
                .format(density_partitian[i - 1], density_partitian[i],
                        *popt_total[i]))
        f.write('\n')
    f.close()


def fit_rad_len_partitian(density_total, rad_len_total, equation_list,
                          density_partitian, outliers, save_path):
    density_water = 1.0
    rad_len_water = 36.083
    popt_total = [None] * (len(density_partitian) + 1)
    pcov_total = [None] * (len(density_partitian) + 1)
    assert len(density_partitian) == len(equation_list) - 1 and len(
        density_partitian) == len(outliers) - 1
    for i in range(len(density_partitian) + 1):
        equation = equation_list[i]
        if i == 0:
            density0 = density_total[np.where(
                density_total <= density_partitian[i])]
            rad_len0 = rad_len_total[np.where(
                density_total <= density_partitian[i])]
            f_rho0 = (density_water * rad_len_water) / (density0 * rad_len0)
            # print(density0, f_rho0, len(density0))
        elif i == len(density_partitian):
            density0 = density_total[np.where(
                density_total > density_partitian[i - 1])]
            rad_len0 = rad_len_total[np.where(
                density_total > density_partitian[i - 1])]
            f_rho0 = (density_water * rad_len_water) / (density0 * rad_len0)
        else:
            density0 = density_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            rad_len0 = rad_len_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            f_rho0 = (density_water * rad_len_water) / (density0 * rad_len0)
        if not np.isinf(outliers[i]):
            if outliers[i] > 0:
                density0 = np.delete(density0, np.where(f_rho0 < outliers[i]))
                f_rho0 = np.delete(f_rho0, np.where(f_rho0 < outliers[i]))
                f_rho0_outlier = f_rho0[np.where(f_rho0 < outliers[i])]
            else:
                density0 = np.delete(density0, np.where(f_rho0 > -outliers[i]))
                f_rho0 = np.delete(f_rho0, np.where(f_rho0 > -outliers[i]))
                f_rho0_outlier = f_rho0[np.where(f_rho0 > -outliers[i])]
        # print(density0, f_rho0)
        # print(density0,f_rho0)
        popt0, pcov0 = curve_fit(equation, density0, f_rho0, method='dogbox')
        popt_total[i] = popt0
        density_fit_range = np.arange(density0.min(), density0.max(), 1e-4)
        plt.figure(1)
        plt.clf()
        plt.plot(density0, f_rho0, 'k+', label='TOPAS')
        plt.plot(density0, equation(density0, *popt0), 'r--', label='Fitting')
        plt.savefig(save_path + '/rad_len_part_{:04d}.png'.format(i))
    return popt_total, pcov_total


def plot_density(topas_data, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    topas_mat_name = topas_data['Material Name']
    density_topas = topas_data['Density']
    density_water = 1.0
    hus = np.arange(-1024, 2001)
    density_topas_ct = np.zeros(len(hus), dtype=np.float32)
    density_calculated = np.zeros(len(hus), dtype=np.float32)

    for i in range(len(topas_mat_name)):
        if 'PatientTissueFromHU' in topas_mat_name[i]:
            mat_name = topas_mat_name[i]
            mat_name = mat_name.replace('PatientTissueFromHU', '')
            if 'Negative' in mat_name:
                mat_name = mat_name.replace('Negative', '')
                hu_value = -1 * int(mat_name)
            else:
                hu_value = int(mat_name)
            assert hu_value >= -1024 and hu_value <= 2000
            density_topas_ct[hu_value + 1024] = density_topas[i]
            density_calculated[hu_value + 1024] = calculate_density(hu_value)
    diff = density_topas_ct - density_calculated
    print(diff.min(), diff.max(), density_topas_ct.max())
    plt.figure(1)
    plt.clf()
    plt.plot(hus, density_topas_ct, 'k', label='TOPAS')
    plt.plot(hus, density_calculated, 'r--', label='Calculated')
    plt.legend(loc='best')
    plt.savefig(save_path + '/density.png')


def fit_rsp(topas_data, topas_mat_data, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    hus = np.arange(-1024, 2001)
    energy_topas = topas_data['Energy']
    rsp_topas = topas_data['SPR (75eV)']
    topas_mat_name = topas_data['Material Name']
    topas_energy = topas_data['Energy']
    topas_mat_name_rho = topas_mat_data['Material Name'].to_numpy()
    density_topas = topas_mat_data['Density'].to_numpy()

    energy_unique = np.unique(energy_topas)
    rsp_topas_ct = np.zeros(len(hus) * len(energy_unique), dtype=np.float32)
    energy_topas_ct = np.zeros(len(hus) * len(energy_unique), dtype=np.float32)
    density_topas_ct = np.zeros(len(hus) * len(energy_unique),
                                dtype=np.float32)
    mat_name_topas_ct = [''] * (len(hus) * len(energy_unique))
    density_water = 1.0
    if os.path.isfile(
            save_path + '/rsp_fitting_rsp_data_0.5.npy') and os.path.isfile(
        save_path +
        '/rsp_fitting_energy_data_0.5.npy') and os.path.isfile(
        save_path +
        '/rsp_fitting_density_data_0.5.npy') and os.path.isfile(
        save_path + '/rsp_fitting_mat_name_data_0.5.npy'):
        with open(save_path + '/rsp_fitting_rsp_data_0.5.npy', 'rb') as f:
            rsp_topas_ct = np.load(f)
        with open(save_path + '/rsp_fitting_energy_data_0.5.npy', 'rb') as f:
            energy_topas_ct = np.load(f)
        with open(save_path + '/rsp_fitting_density_data_0.5.npy', 'rb') as f:
            density_topas_ct = np.load(f)
        with open(save_path + '/rsp_fitting_mat_name_data_0.5.npy', 'rb') as f:
            mat_name_topas_ct = np.load(f)

    else:
        for i in trange(len(topas_mat_name)):
            if 'PatientTissueFromHU' in topas_mat_name[i]:
                mat_name = topas_mat_name[i]
                mat_name = mat_name.replace('PatientTissueFromHU', '')
                if 'Negative' in mat_name:
                    mat_name = mat_name.replace('Negative', '')
                    hu_value = -1 * int(mat_name)
                else:
                    hu_value = int(mat_name)
                # if hu_value == -1001:
                #     continue
                assert hu_value >= -1024 and hu_value <= 2000
                assert energy_topas[i] >= 0.01 and energy_topas[
                    i] <= 351.0, energy_topas[i]
                # rsp_topas_ct[hu_value + 1000] = rsp_topas_ek[i]
                HUbin = hu_value + 1024
                Ebin = np.where(energy_topas[i] == energy_unique)[0]
                assert len(Ebin) == 1
                Ebin = Ebin[0]
                ind = HUbin + Ebin * len(hus)
                energy_topas_ct[ind] = energy_topas[i]
                rsp_topas_ct[ind] = rsp_topas[i]
                density_topas_ct[ind] = density_topas[
                    topas_mat_name_rho == topas_mat_name[i].strip()][0]
                mat_name_topas_ct[ind] = topas_mat_name[i].strip()
                # print(topas_mat_name[i].strip(), mat_name_topas_ct[ind])
            elif 'G4_WATER' == topas_mat_name[i]:
                pass
                # rsp_water = rsp_topas[i]
                # density_water = density_topas['G4_WATER']
                # print(density_water)
        with open(save_path + '/rsp_fitting_rsp_data_0.5.npy', 'wb') as f:
            np.save(f, rsp_topas_ct)
        with open(save_path + '/rsp_fitting_energy_data_0.5.npy', 'wb') as f:
            np.save(f, energy_topas_ct)
        with open(save_path + '/rsp_fitting_density_data_0.5.npy', 'wb') as f:
            np.save(f, density_topas_ct)
        with open(save_path + '/rsp_fitting_mat_name_data_0.5.npy', 'wb') as f:
            np.save(f, mat_name_topas_ct)
        sys.exit()
    rsp_hu0 = rsp_topas_ct[mat_name_topas_ct == 'PatientTissueFromHU0']
    energy_hu0 = energy_topas_ct[mat_name_topas_ct == 'PatientTissueFromHU0']
    fs_rho_total = density_water * rsp_topas_ct / density_topas_ct
    rsp_test = rsp_topas_ct[energy_topas_ct == 100.0]
    density_test = density_topas_ct[energy_topas_ct == 100.0]
    fs_rho_test = density_water * rsp_test / density_test
    plt.figure(1)
    plt.clf()
    plt.plot(energy_hu0, rsp_hu0, 'k.')
    plt.title('Relative stopping power of HU:0')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Relative stopping power to water')
    plt.savefig(save_path + '/rsp_hu0.png')

    plt.figure(1)
    plt.clf()
    plt.plot(density_test, fs_rho_test, 'k.')
    plt.title('fs at E:100MeV')
    plt.xlabel('Density (g/cm3)')
    plt.ylabel('fs')
    plt.savefig(save_path + '/fs_100MeV.png')

    density_partitian = np.array([
        0.095, 0.902, 0.937, 0.96, 0.984, 1.0, 1.0175, 1.08, 1.1035, 1.15, 1.21,
        1.28, 1.34, 1.41, 1.48, 1.54, 1.61, 1.68, 1.74, 1.81, 1.88, 1.95, 2.01
    ])

    print(density_partitian)
    equation_list = [eq0_rsp] * (len(density_partitian) + 1)
    outliers = [False] * (len(density_partitian) + 1)
    outliers[0:3] = [True, True, True]
    outliers[8] = True
    energies = np.arange(1, 100, 10)

    plot_rsp_partitian(energy_topas_ct, density_topas_ct, rsp_topas_ct,
                       energies, density_partitian, outliers, save_path)
    popt_total, pcov_total = fit_rsp_per_energy_partitian(
        energy_topas_ct, density_topas_ct, rsp_topas_ct, energies,
        equation_list, density_partitian, outliers, save_path)
    density_water = 1.0
    density_hu0 = calculate_density(0)
    print(density_hu0)
    fit_ind = np.max(np.where(density_partitian < density_hu0))
    coefficient = popt_total[fit_ind + 1]
    eq = equation_list[fit_ind + 1]
    print(fit_ind)
    plt.figure(1)
    plt.clf()
    plt.plot(energy_hu0, rsp_hu0 * 1.0 / density_hu0, 'r+', label='TOPAS')
    for e in range(1, 352, 1):
        plt.plot(e, eq(e, *coefficient), 'k+')
    plt.plot([], [], 'k+', label='fitted')
    plt.legend(loc='best')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('fs')
    plt.savefig(save_path + '/hu0_fs.png')

    fit_ind = np.max(np.where(np.array(density_partitian) < density_water))
    # print(fit_ind)
    coefficient = popt_total[fit_ind + 1]
    eq = equation_list[fit_ind + 1]
    plt.figure(1)
    plt.clf()
    plt.plot(energy_hu0, rsp_hu0, 'r+', label='TOPAS')
    for e in range(1, 350, 1):
        plt.plot(e, eq(e, *coefficient), 'k+')
    plt.plot([], [], 'k+', label='fitted')
    plt.legend(loc='best')
    plt.xlabel('Energy (MeV)')
    plt.ylabel('fs')
    plt.savefig(save_path + '/water_fs.png')

    sample_ek = 100
    sample_densities = np.arange(0.00121, np.max(density_test), 0.01)
    sample_fs = np.zeros(len(sample_densities), dtype=np.float32)
    sample_rsp = np.zeros(len(sample_densities), dtype=np.float32)
    for i in range(len(sample_densities)):
        if sample_densities[i] < density_partitian[0]:
            fit_ind = 0
        elif sample_densities[i] > density_partitian[-1]:
            fit_ind = len(density_partitian)
        else:
            fit_ind = np.max(
                np.where(density_partitian < sample_densities[i])) + 1
        # print(fit_ind)
        coefficient = popt_total[fit_ind]
        eq = equation_list[fit_ind]
        fs0 = eq(sample_ek, *coefficient)
        rsp0 = fs0 * sample_densities[i] / density_water
        sample_fs[i] = fs0
        sample_rsp[i] = rsp0

    sample_densities1 = np.copy(sample_densities)
    sample_densities2 = np.copy(sample_densities)
    discontinue_ind = np.array(
        np.where(np.abs(sample_fs[1:] - sample_fs[:-1]) > 0.02)[0]) + 1
    sample_densities1 = np.insert(sample_densities1, list(discontinue_ind),
                                  np.nan)
    sample_densities2 = np.insert(sample_densities2, list(discontinue_ind),
                                  np.nan)
    sample_fs = np.insert(sample_fs, list(discontinue_ind), np.nan)
    sample_rsp = np.insert(sample_rsp, list(discontinue_ind), np.nan)

    plt.figure(1)
    plt.clf()
    plt.plot(density_test[::40],
             fs_rho_test[::40],
             'k+',
             label='TOPAS',
             markersize=8)
    plt.plot(sample_densities1, sample_fs, 'r--', label='Fitted')
    plt.xlabel('Densities')
    plt.ylabel('fs')
    plt.legend(loc='best')
    plt.savefig(save_path + '/sample_fs_100MeV.png')

    plt.figure(1)
    plt.clf()
    plt.plot(density_test[::40],
             rsp_test[::40],
             'k+',
             label='TOPAS',
             markersize=8)
    plt.plot(sample_densities2, sample_rsp, 'r--', label='Fitted')
    plt.xlabel('Densities (g/cm3)')
    plt.ylabel('Relative stopping power to water')
    plt.title(
        'Fitting results for density to relative stopping power conversion at 100MeV'
    )
    plt.legend(loc='best')
    plt.savefig(save_path + '/sample_rsp_100MeV.png')

    f = open(save_path + '/fs_fit.txt', 'w')
    for i in range(len(equation_list)):
        if i == 0:
            # print(
            #     'if(density_tmp<={:4f}){{fs={:.4f} +  {:.4e}*mqi::mqi_ln(Ek+{:.4e})+{:.4e}*mqi::mqi_ln(Ek*Ek+{:.4e});}}'
            #     .format(density_partitian[i], *popt_total[i]))
            f.write((
                'if(density_tmp<={:4f}){{fs={:.4f} +  {:.4e}*logf(Ek+{:.4e})+{:.4e}*logf(Ek*Ek+{:.4e});}}'
                .format(density_partitian[i], *popt_total[i])))

        elif i == len(equation_list) - 1:
            # print(
            #     'else if(density_tmp>{:.4f}){{fs={:.4f} + {:.4e} *mqi::mqi_ln(Ek + {:.4e})+{:.4e}*mqi::mqi_ln(Ek*Ek+{:.4e});}}'
            #     .format(density_partitian[i - 1], *popt_total[i]))
            f.write(
                'else if(density_tmp>{:.4f}){{fs={:.4f} + {:.4e} *logf(Ek + {:.4e})+{:.4e}*logf(Ek*Ek+{:.4e});}}'
                .format(density_partitian[i - 1], *popt_total[i]))
        else:
            # print(
            #     'else if(density_tmp<={:4f}){{fs={:.4f} +  {:.4e}*mqi::mqi_ln(Ek+{:.4e})+{:.4e}*mqi::mqi_ln(Ek*Ek+{:.4e});}}'
            #     .format(density_partitian[i], *popt_total[i]))
            f.write(
                'else if(density_tmp<={:4f}){{fs={:.4f} +  {:.4e}*logf(Ek+{:.4e})+{:.4e}*logf(Ek*Ek+{:.4e});}}'
                .format(density_partitian[i], *popt_total[i]))
        f.write('\n')
    f.close()


def select_color_and_marker(count):
    color_array = ['k', 'r', 'b', 'c', 'm', 'g', 'y']
    marker_array = ['^', 'p', 's', 'd']
    line_array = ['--', '-.', ':', '-']
    assert count < len(color_array) * len(marker_array)
    color = color_array[count % len(color_array)]
    marker = marker_array[int(count / len(color_array))]
    line_style = line_array[int(count / len(color_array))]
    for_data = color + marker
    for_fit = color + line_style
    return for_data, for_fit


def plot_rsp_partitian(energy_total, density_total, rsp_total, energies_plot,
                       density_partitian, outliers, save_path):
    density_water = 1.0
    energy_range = np.arange(0.5, 350.5, 0.5)
    energy_range = np.append([0.01], energy_range)
    for i in range(len(density_partitian) + 1):
        if i == 0:
            density0 = density_total[np.where(
                density_total <= density_partitian[i])]
            rsp0 = rsp_total[np.where(density_total <= density_partitian[i])]
            energy0 = energy_total[np.where(
                density_total <= density_partitian[i])]
            fs_rho0 = density_water * rsp0 / density0
        elif i == len(density_partitian):
            density0 = density_total[np.where(
                density_total > density_partitian[i - 1])]
            rsp0 = rsp_total[np.where(
                density_total > density_partitian[i - 1])]
            energy0 = energy_total[np.where(
                density_total > density_partitian[i - 1])]
            fs_rho0 = density_water * rsp0 / density0
        else:
            density0 = density_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            rsp0 = rsp_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            energy0 = energy_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            fs_rho0 = density_water * rsp0 / density0

        energy_tmp = np.zeros_like(energy0)
        fs_rho_tmp = np.zeros_like(fs_rho0)
        density_tmp = np.zeros_like(density0)
        count = 0
        if outliers[i]:
            for e in range(len(energy_range)):
                energy_select = energy0[np.where(energy0 == energy_range[e])]
                fs_rho_select = fs_rho0[np.where(energy0 == energy_range[e])]
                density_select = density0[np.where(energy0 == energy_range[e])]

                remove_ind = np.where(
                    np.abs(fs_rho_select - np.median(fs_rho_select)) > 1e-3)
                energy1 = np.delete(energy_select, remove_ind)
                fs_rho1 = np.delete(fs_rho_select, remove_ind)
                density1 = np.delete(density_select, remove_ind)
                assert len(energy1) == len(fs_rho1) and len(energy1) == len(
                    density1)
                energy_tmp[count:count + len(energy1)] = energy1
                fs_rho_tmp[count:count + len(fs_rho1)] = fs_rho1
                density_tmp[count:count + len(density1)] = density1
                count += len(energy1)
            energy0 = energy_tmp[:count]
            fs_rho0 = fs_rho_tmp[:count]
            density0 = density_tmp[:count]
        plt.figure(1)
        plt.clf()
        ii = 0
        for e in energies_plot:
            color_and_marker_data, color_and_marker_fit = select_color_and_marker(
                ii)
            density_e1 = density0[np.where(energy0 == e)]
            fs_rho_e1 = fs_rho0[np.where(energy0 == e)]
            plt.plot(density_e1,
                     fs_rho_e1,
                     color_and_marker_data,
                     label='TOPAS E: {:.1f}'.format(e))
            ii += 1
        plt.savefig(save_path + '/rsp_part_{:03d}.png'.format(i))


def fit_rsp_per_energy_partitian(energy_total, density_total, rsp_total,
                                 energies_plot, equation_list,
                                 density_partitian, outliers, save_path):
    density_water = 1.0
    popt_total = [None] * (len(density_partitian) + 1)
    pcov_total = [None] * (len(density_partitian) + 1)
    energy_range = np.arange(0.5, 350.5, 0.5)
    energy_range = np.append([0.01], energy_range)
    assert len(density_partitian) == len(equation_list) - 1 and len(
        density_partitian) == len(outliers) - 1
    bounds = ([
                  -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf
              ], np.inf)
    for i in range(len(density_partitian) + 1):
        equation = equation_list[i]
        if i == 0:
            density0 = density_total[np.where(
                density_total <= density_partitian[i])]
            rsp0 = rsp_total[np.where(density_total <= density_partitian[i])]
            energy0 = energy_total[np.where(
                density_total <= density_partitian[i])]
            fs_rho0 = density_water * rsp0 / density0

        elif i == len(density_partitian):
            density0 = density_total[np.where(
                density_total > density_partitian[i - 1])]
            rsp0 = rsp_total[np.where(
                density_total > density_partitian[i - 1])]
            energy0 = energy_total[np.where(
                density_total > density_partitian[i - 1])]
            fs_rho0 = density_water * rsp0 / density0
        else:
            density0 = density_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            rsp0 = rsp_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            energy0 = energy_total[np.where(
                (density_total > density_partitian[i - 1])
                & (density_total <= density_partitian[i]))]
            fs_rho0 = density_water * rsp0 / density0
        energy_tmp = np.zeros_like(energy0)
        fs_rho_tmp = np.zeros_like(fs_rho0)
        density_tmp = np.zeros_like(density0)
        count = 0
        if outliers[i]:
            for e in range(len(energy_range)):
                energy_select = energy0[np.where(energy0 == energy_range[e])]
                fs_rho_select = fs_rho0[np.where(energy0 == energy_range[e])]
                density_select = density0[np.where(energy0 == energy_range[e])]

                remove_ind = np.where(
                    np.abs(fs_rho_select - np.median(fs_rho_select)) > 1e-3)
                energy1 = np.delete(energy_select, remove_ind)
                fs_rho1 = np.delete(fs_rho_select, remove_ind)
                density1 = np.delete(density_select, remove_ind)
                assert len(energy1) == len(fs_rho1) and len(energy1) == len(
                    density1)
                energy_tmp[count:count + len(energy1)] = energy1
                fs_rho_tmp[count:count + len(fs_rho1)] = fs_rho1
                density_tmp[count:count + len(density1)] = density1
                count += len(energy1)
            energy0 = energy_tmp[:count]
            fs_rho0 = fs_rho_tmp[:count]
            density0 = density_tmp[:count]

        fs_energy = np.zeros(len(energy_range), dtype=np.float32)
        density_range = np.zeros_like(energy_range)

        for j in range(len(energy_range)):
            ind = np.where(energy0 == energy_range[j])
            fs_rho1 = fs_rho0[ind]
            if not len(
                    np.where(np.abs(fs_rho1 - np.median(fs_rho1)) < 1e-3)
                    [0]) > 0.95 * len(fs_rho1):
                print('i {:d} {:d} {:d}'.format(
                    i,
                    len(
                        np.where(
                            np.abs(fs_rho1 - np.median(fs_rho1)) < 1e-3)[0]),
                    len(fs_rho1)))
                plt.figure(1)
                plt.clf()
                plt.plot(fs_rho1)
                plt.show()
            assert len(
                np.where(np.abs(fs_rho1 - np.median(fs_rho1)) < 1e-3)[0]
            ) > 0.95 * len(fs_rho1), 'i {:d} {:d} {:d}'.format(
                i,
                len(np.where(np.abs(fs_rho1 - np.median(fs_rho1)) < 1e-3)[0]),
                len(fs_rho1))

            fs_energy[j] = np.median(fs_rho1)
        if i == 0 or i == 4 or i == 5 or i == 13 or i == 14:
            p0 = [0.7, 0.1, 0, 1, 1, 1, 1, 1]
            bounds = ([
                          -np.inf, -np.inf, -energy_range[0] + 0.0001, -np.inf,
                                            -energy_range[0] ** 2 + 0.0001, -np.inf, -np.inf, -np.inf
                      ], np.inf)
        # elif i == 13:
        #     p0 = [1.0, 0.03, 0.03, -0.01, 4, 1, 1, 1]
        #     bounds = ([
        #         -np.inf, -np.inf, -energy_range[0] + 0.0001, -np.inf,
        #         -energy_range[0]**2 + 0.0001, -np.inf, -np.inf, -np.inf
        #     ], np.inf)
        else:
            p0 = [1, 1, 0, 1, 1, 1, 1, 1]
            bounds = ([
                          -np.inf, -np.inf, -energy_range[0] + 0.0001, -np.inf,
                                            -energy_range[0] ** 2 + 0.0001, -np.inf, -np.inf, -np.inf
                      ], np.inf)

        popt0, pcov0 = curve_fit(equation,
                                 energy_range,
                                 fs_energy,
                                 p0=p0,
                                 bounds=bounds,
                                 maxfev=10000)
        # print(popt0)
        popt_total[i] = popt0

        plt.figure(1)
        plt.clf()
        plt.figure(1)
        plt.clf()
        plt.plot(energy_range, fs_energy, 'k.', label='TOPAS')
        plt.plot(energy_range,
                 equation(energy_range, *popt0),
                 '.r',
                 label='Fitted')
        plt.legend(loc='best')
        plt.xlabel('Energy (MeV)')
        plt.ylabel('fs(density)')
        if i == 0:
            plt.title('fs(density) fitting for density <= {:.5f}'.format(
                density_partitian[i]))
        elif i == len(density_partitian):
            plt.title('fs(density) fitting for {:.5f} < density'.format(
                density_partitian[i - 1]))
        else:
            plt.title(
                'fs(density) fitting for {:.5f} < density <= {:.5f}'.format(
                    density_partitian[i - 1], density_partitian[i]))

        plt.savefig(save_path +
                    '/rsp_per_energy_fit_part_{:03d}.png'.format(i))

    return popt_total, pcov_total


save_path = './fitting'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
path = './'
rsp_topas_total = pd.read_csv(path + '/SPR_cor.csv')
mat_data_ray = pd.read_csv(path + '/MaterialInfo_cor.csv')
plt.rcParams["font.family"] = "Times New Roman"

# plot_density(mat_data_ray, save_path + '/density')
# fit_rad_len(mat_data_ray, save_path + '/rad_len')
fit_rsp(rsp_topas_total, mat_data_ray, save_path + '/rsp')
