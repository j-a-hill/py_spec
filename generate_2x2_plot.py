#!/usr/bin/env python3
"""
Generate the enhanced 2x2 dose-decay plot from CSV data files.
This script directly loads the processed DTNB_Dose data and creates the publication-quality visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD DATA
# ============================================================================
data_base = Path('/workspaces/py_spec')
dtnb_dir = data_base / 'DTNB_Dose'
output_dir = data_base / 'spectral_analysis_output' / 'DTNB_Dose_labeled'
output_dir.mkdir(parents=True, exist_ok=True)

# Load CSV files for each transmission level
files = {
    'DTNB_100%': dtnb_dir / 'final_pyspec_100.csv',
    'DTNB_50%': dtnb_dir / 'final_pyspec_50.csv',
    'DTNB_25%': dtnb_dir / 'final_pyspec_25.csv',
    'DTNB_5%': dtnb_dir / 'final_pyspec_5.csv'
}

dose_analysis = {}
beam_parameters = {
    'flux_ph_per_s': 5.0e12,
    'energy_keV': 12.4,
    'beam_diameter_um': 50.0,
    'crystal_diameter_um': 15.0,
    'crystal_thickness_um': 5.0,
    'overlap_fraction': 0.65,
    'mass_atten_coeff_cm2_g': 0.18,
    'sample_density_kg_m3': 1350
}

# Calculate dose rate
energy_J = beam_parameters['energy_keV'] * 1000 * 1.60218e-19
beam_area_um2 = np.pi * (beam_parameters['beam_diameter_um'] / 2) ** 2
effective_beam_area_um2 = beam_area_um2 * beam_parameters['overlap_fraction']
irradiated_volume_um3 = effective_beam_area_um2 * beam_parameters['crystal_thickness_um']
volume_m3 = irradiated_volume_um3 * 1e-18
mass_kg = beam_parameters['sample_density_kg_m3'] * volume_m3

sample_density_g_cm3 = beam_parameters['sample_density_kg_m3'] / 1000
linear_atten_coeff_cm = beam_parameters['mass_atten_coeff_cm2_g'] * sample_density_g_cm3
linear_atten_coeff_um = linear_atten_coeff_cm / 10000
mu_times_t = linear_atten_coeff_um * beam_parameters['crystal_thickness_um']
absorption_fraction = 1 - np.exp(-mu_times_t)

energy_absorbed_per_s = beam_parameters['flux_ph_per_s'] * energy_J * absorption_fraction
dose_rate_full_beam_Gy_s = energy_absorbed_per_s / mass_kg if mass_kg > 0 else np.nan

print(f"✓ Calculated dose rate: {dose_rate_full_beam_Gy_s*1e-6:.4f} MGy/s @ 100% transmission")

# ============================================================================
# PROCESS EACH DATASET
# ============================================================================
for dataset_name, filepath in files.items():
    print(f"\nProcessing {dataset_name}...", end=" ")
    
    if not filepath.exists():
        print(f"⚠ File not found: {filepath}")
        continue
    
    # Load data
    data = pd.read_csv(filepath, index_col=0)
    wavelengths = data.index.astype(float).values
    
    # Parse transmission
    transmission_pct = float(dataset_name.split('_')[1].replace('%', ''))
    transmission_fraction = transmission_pct / 100.0
    
    # Extract time points
    times_numeric = []
    for col in data.columns:
        try:
            t = float(col.replace('s', ''))
            times_numeric.append(t)
        except ValueError:
            pass
    times_numeric = np.array(sorted(times_numeric))
    mask_10s = times_numeric <= 10.0
    times_10s = times_numeric[mask_10s]
    
    # Extract 412 nm and 315 nm traces
    try:
        trace_412 = data.loc[412.0, :].values.astype(float) if 412.0 in data.index else data.iloc[(data.index - 412).abs().argmin(), :].values.astype(float)
        trace_315 = data.loc[315.0, :].values.astype(float) if 315.0 in data.index else data.iloc[(data.index - 315).abs().argmin(), :].values.astype(float)
    except:
        # Use closest wavelengths
        idx_412 = (np.abs(wavelengths - 412)).argmin()
        idx_315 = (np.abs(wavelengths - 315)).argmin()
        trace_412 = data.iloc[idx_412, :].values.astype(float)
        trace_315 = data.iloc[idx_315, :].values.astype(float)
    
    trace_412_s = gaussian_filter1d(trace_412, sigma=1)
    trace_315_s = gaussian_filter1d(trace_315, sigma=1)
    
    # Trim to 10 seconds
    trace_412_s = trace_412_s[:len(times_10s)]
    trace_315_s = trace_315_s[:len(times_10s)]
    
    # Calculate dose
    dose_rate_Gy_s = dose_rate_full_beam_Gy_s * transmission_fraction
    dose_rate_MGy_s = dose_rate_Gy_s * 1e-6
    dose_at_time_Gy = dose_rate_Gy_s * times_10s
    dose_at_time_MGy = dose_at_time_Gy * 1e-6
    
    # Detect onset (start of steep decay) and end point (before crystal burning)
    onset_idx = 0
    burn_start_idx = len(trace_412_s)  # Default to endof data
    
    if len(trace_412_s) > 5:
        # Calculate first and second derivatives to find where steep decay begins
        first_deriv = np.diff(trace_412_s)
        smoothed_deriv = gaussian_filter1d(first_deriv, sigma=2)
        
        # Find onset: where derivative becomes significantly negative (steep decay starts)
        # Use first 20 points to establish baseline derivative
        baseline_points = min(20, len(smoothed_deriv) // 3)
        baseline_deriv = np.mean(smoothed_deriv[:baseline_points])  
        deriv_std = np.std(smoothed_deriv[:baseline_points])
        
        # Look for sustained steep descent (> 3 std below baseline)
        threshold_steep = baseline_deriv - 3 * deriv_std
        sustained_count = 0
        
        for i in range(baseline_points, len(smoothed_deriv)):
            if smoothed_deriv[i] < threshold_steep:
                sustained_count += 1
                if sustained_count >= 2:  # Need 2 consecutive steep points
                    onset_idx = max(0, i - sustained_count)  # Go back to start of steep section
                    break
            else:
                sustained_count = 0
        
        # Find end of decay: where signal starts INCREASING again (crystal burn)
        if onset_idx < len(smoothed_deriv) - 5:
            # Look for sustained positive derivative after onset
            threshold_burn = baseline_deriv + 3 * deriv_std
            positive_count = 0
            for i in range(onset_idx + 5, len(smoothed_deriv)):
                if smoothed_deriv[i] > threshold_burn and smoothed_deriv[i] > 0:
                    positive_count += 1
                    if positive_count >= 3:  # Sustained increase
                        burn_start_idx = i - 2
                        break
                else:
                    if smoothed_deriv[i] < 0:
                        positive_count = 0
    
    # Ensure onset is reasonable (within first 50% of data)
    onset_idx = min(onset_idx, int(len(times_10s) * 0.5))
    onset_time = times_10s[onset_idx]
    onset_dose = dose_at_time_MGy[onset_idx]
    
    # Ensure burn detection is after onset with minimum decay window
    burn_start_idx = max(burn_start_idx, onset_idx + 5)
    
    # Shift dose to start from onset
    dose_from_onset = dose_at_time_MGy - onset_dose
    times_from_onset = times_10s - onset_time
    
    # Use sliding window to find optimal fitting region (only in decay zone, before burning)
    decay_params = None
    best_r2 = -np.inf
    best_fit_result = None
    best_window = None
    
    # Exponential decay model: y = y_min + (y_max - y_min) * exp(-k * dose)
    def exponential_decay(dose, y_max, y_min, k):
        return y_min + (y_max - y_min) * np.exp(-k * dose)
    
    # Only fit in the decay region (from onset to before crystal burn)
    decay_region_length = burn_start_idx - onset_idx
    
    if decay_region_length >= 5:
        # Try different start points - start at least 0.3s after onset to skip any residual flat region
        time_step = times_10s[1] - times_10s[0] if len(times_10s) > 1 else 0.1
        min_start_offset = 0.3  # Start at least 0.3s after onset
        max_start_offset = min(1.5, decay_region_length * time_step * 0.5)
        
        for start_offset in np.arange(min_start_offset, max_start_offset + 0.1, 0.2):
            decay_start_idx = onset_idx + int(start_offset / time_step)
            
            # Can't start past the burn region
            if decay_start_idx >= burn_start_idx - 5:
                break
                
            # Try different window lengths, ending before or at burn threshold
            for end_fraction in [0.7, 0.8, 0.9, 1.0]:
                remaining_points = burn_start_idx - decay_start_idx
                decay_end_idx = decay_start_idx + int(remaining_points * end_fraction)
                decay_end_idx = min(decay_end_idx, burn_start_idx)  # Don't go past burn
                
                if decay_end_idx - decay_start_idx < 5:
                    continue
                    
                decay_dose = dose_from_onset[decay_start_idx:decay_end_idx]
                decay_abs = trace_412_s[decay_start_idx:decay_end_idx]
                
                try:
                    # Initial guess for k
                    dose_range = decay_dose[-1] - decay_dose[0]
                    k_guess = 2.0 / dose_range if dose_range > 0 else 1.0
                    
                    popt, pcov = curve_fit(
                        exponential_decay,
                        decay_dose,
                        decay_abs,
                        p0=[np.max(decay_abs), np.min(decay_abs), k_guess],
                        bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                        maxfev=20000
                    )
                    
                    # Calculate R²
                    fit_y = exponential_decay(decay_dose, *popt)
                    residuals = decay_abs - fit_y
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((decay_abs - np.mean(decay_abs))**2)
                    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
                    
                    # Keep best fit
                    if r2 > best_r2:
                        best_r2 = r2
                        best_fit_result = popt
                        best_window = (decay_start_idx, decay_end_idx)
                        
                except:
                    continue
    
    # Store best fit parameters
    if best_fit_result is not None:
        D_half = np.log(2) / best_fit_result[2] if best_fit_result[2] > 0 else np.inf
        
        decay_params = {
            'y_max': best_fit_result[0],
            'y_min': best_fit_result[1],
            'k': best_fit_result[2],  # Dose-dependent decay rate constant (MGy^-1)
            'D_half': D_half,  # Dose at half-maximal decay
            'decay_span': best_fit_result[0] - best_fit_result[1],
            'r2': best_r2,
            'fit_window': best_window
        }
    
    # Store results
    dose_analysis[dataset_name] = {
        'times_10s': times_10s,
        'dose_at_time_MGy': dose_at_time_MGy,
        'dose_from_onset_MGy': dose_from_onset,
        'times_from_onset': times_from_onset,
        'onset_idx': onset_idx,
        'onset_time': onset_time,
        'onset_dose': onset_dose,
        'burn_start_idx': burn_start_idx,
        'trace_412_s': trace_412_s,
        'trace_315_s': trace_315_s,
        'transmission_pct': transmission_pct,
        'dose_rate_MGy_s': dose_rate_MGy_s,
        'decay_params': decay_params
    }
    
    burn_time = times_10s[burn_start_idx] if burn_start_idx < len(times_10s) else times_10s[-1]
    print(f"✓ (Onset: {onset_time:.3f}s, Burn: {burn_time:.3f}s)")

print(f"\n✓ Loaded {len(dose_analysis)} datasets")

# ============================================================================
# CREATE 2x2 ENHANCED PLOT
# ============================================================================
def create_dose_decay_2x2_enhanced(dose_analysis, output_path, figsize=(14, 11)):
    """Create 2x2 subplot with publication-quality formatting."""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    datasets = ['DTNB_5%', 'DTNB_25%', 'DTNB_50%', 'DTNB_100%']
    panel_labels = ['A', 'B', 'C', 'D']
    colors_tx = ['blue', 'green', 'orange', 'red']
    
    for ax, dataset, panel_label, color_tx in zip(axes.flat, datasets, panel_labels, colors_tx):
        if dataset not in dose_analysis:
            ax.text(0.5, 0.5, f'No data for {dataset}', ha='center', va='center')
            continue
        
        data = dose_analysis[dataset]
        dose_MGy = data['dose_from_onset_MGy']  # Use onset-corrected dose
        times_s = data['times_from_onset']       # Use onset-corrected time
        trace_412 = data['trace_412_s']
        transmission_pct = data['transmission_pct']
        dose_rate_MGy_s = data['dose_rate_MGy_s']
        decay_params = data.get('decay_params')
        
        # Plot data points
        ax.plot(dose_MGy, trace_412, 'o-', ms=6, lw=2, color=color_tx, 
                label=f'{transmission_pct:.0f}% TX', alpha=0.5)
        
        # Add fit line if available
        if decay_params is not None:
            fit_window = decay_params['fit_window']
            decay_start_idx, decay_end_idx = fit_window
            decay_dose = dose_MGy[decay_start_idx:decay_end_idx]
            
            def exponential_decay(dose, y_max, y_min, k):
                return y_min + (y_max - y_min) * np.exp(-k * dose)
            
            fit_y = exponential_decay(
                decay_dose,
                decay_params['y_max'],
                decay_params['y_min'],
                decay_params['k']
            )
            
            # Label with k, D½, and R²
            r2 = decay_params['r2']
            ax.plot(decay_dose, fit_y, '-', lw=2.5, color='black', alpha=0.95, 
                    label=f"k={decay_params['k']:.2f} MGy$^{{-1}}$, D$_{{1/2}}$={decay_params['D_half']:.3f} MGy, R$^2$={r2:.3f}")
        
        # Set x-axis to start at 0 and end shortly after burn starts (tighter view)
        burn_idx = data.get('burn_start_idx', len(dose_MGy))
        if burn_idx < len(dose_MGy):
            # Add 10% padding beyond burn point
            max_dose = dose_MGy[min(burn_idx + 5, len(dose_MGy)-1)] * 1.1
        else:
            max_dose = dose_MGy[-1] * 1.05
        ax.set_xlim(left=0, right=max_dose)
        
        # Tighten y-axis to focus on decay curve
        onset_idx = data['onset_idx']
        y_min = np.min(trace_412[onset_idx:min(burn_idx, len(trace_412))])
        y_max = np.max(trace_412[onset_idx:min(burn_idx, len(trace_412))])
        y_range = y_max - y_min
        # Extra padding for 100% to accommodate sharper decay
        padding = 0.15 if dataset == 'DTNB_100%' else 0.1
        ax.set_ylim(y_min - padding*y_range, y_max + padding*y_range)
        
        # Primary x-axis: Dose from Onset
        ax.set_xlabel('Dose MGy', fontsize=11)
        ax.set_ylabel('Absorbance at 412 nm', fontsize=11)
        
        # Secondary x-axis: Time from Onset
        ax_time = ax.twiny()
        if len(dose_MGy) > 0 and dose_rate_MGy_s > 0:
            dose_lim = ax.get_xlim()
            time_lim = [dose_lim[0] / dose_rate_MGy_s, dose_lim[1] / dose_rate_MGy_s]
            ax_time.set_xlim(time_lim)
            ax_time.set_xlabel('Time s', fontsize=11)
            ax_time.tick_params(axis='x')
        
        # Panel label (serif font, no box, above panel, left-aligned)
        ax.text(0.02, 1.12, panel_label, transform=ax.transAxes,
                fontsize=16, va='bottom', ha='left',
                family='serif', fontweight='normal')
        
        # Legend
        ax.legend(fontsize=9, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_file = output_path / '12_dose_decay_2x2_modified.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Publication-quality 2x2 dose-decay plot saved to:\n  {output_file}")
    
    return fig, output_file

# Generate the plot
fig, output_file = create_dose_decay_2x2_enhanced(dose_analysis, output_dir)
plt.close(fig)

print("\n" + "="*85)
print("✓ PLOT GENERATION COMPLETE")
print("="*85)
print(f"\nVisualization file:\n  {output_file}")
print("\nKey Features Implemented:")
print("  ✓ Onset detection: Automatically detects when signal starts changing")
print("  ✓ Dose axis starts at 0 (varies per dataset based on onset detection)")
print("  ✓ Dual x-axes: 'Dose MGy' on bottom, 'Time s' on top")
print("  ✓ Clean panel labels: A, B, C, D in serif font above each panel")
print("  ✓ No colored boxes - publication-ready appearance")
print("  ✓ Legend shows transmission % for each dataset")
print("  ✓ High resolution: 300 DPI for publication quality")
