"""
Generate a 2x2 subplot figure showing dose-dependent decay for DTNB samples.
This creates the modified version with:
- Panels labeled A-D
- Dual axes: dose (MGy) and time
- Simplified units (just "MGy")
- Legend showing transmission %
- Kinetics information
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# This script assumes dose_analysis dict is available from labeled_spec_analysis.ipynb
# dose_analysis contains keys: 'DTNB_100%', 'DTNB_50%', 'DTNB_25%', 'DTNB_5%'

def create_dose_decay_2x2(dose_analysis, output_path='dose_decay_2x2.png', figsize=(14, 11)):
    """
    Create a 2x2 subplot figure of dose-dependent decay with dual axes.
    
    Parameters:
    -----------
    dose_analysis : dict
        Dictionary containing dose analysis data for each dataset
    output_path : str
        Path to save the figure
    figsize : tuple
        Figure size (width, height)
    """
    
    # Define the 4 datasets and their panel labels
    datasets = ['DTNB_100%', 'DTNB_50%', 'DTNB_25%', 'DTNB_5%']
    panel_labels = ['A', 'B', 'C', 'D']
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # Red, Orange, Green, Blue
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for idx, (dataset_name, panel_label, color, ax) in enumerate(zip(datasets, panel_labels, colors, axes)):
        if dataset_name not in dose_analysis:
            ax.text(0.5, 0.5, f'{dataset_name} not found', ha='center', va='center')
            continue
            
        dose_data = dose_analysis[dataset_name]
        
        # Extract data
        dose_MGy = np.array(dose_data['dose_at_time_MGy'])
        trace_412 = np.array(dose_data['trace_412_s'])
        times = np.array(dose_data['times_10s'])
        transmission_pct = dose_data['transmission_pct']
        sig_params = dose_data.get('sigmoid_params')
        decay_k = dose_data.get('decay_k_MGy_inv', np.nan)
        decay_r2 = dose_data.get('decay_r2', np.nan)
        decay_start_MGy = dose_data.get('decay_start_MGy', np.nan)
        
        # Plot data points
        ax.scatter(dose_MGy, trace_412, s=80, alpha=0.7, color=color, 
                  edgecolors='black', linewidth=1, zorder=3, label='Data')
        
        # Add exponential/sigmoid fit if available
        if sig_params and not np.isnan(decay_k):
            # Create fit curve for decay window
            decay_start_idx = np.searchsorted(dose_MGy, decay_start_MGy)
            if decay_start_idx < len(dose_MGy):
                decay_dose = dose_MGy[decay_start_idx:]
                
                # Sigmoid function
                def sigmoid(x, IC50, n, max_val, min_val):
                    return min_val + (max_val - min_val) / (1 + np.exp(n * (IC50 - x)))
                
                IC50 = sig_params.get('IC50', decay_k)
                n = sig_params.get('n', 1.0)
                max_val = trace_412.max()
                min_val = trace_412.min()
                
                fit_curve = sigmoid(decay_dose, IC50, n, max_val, min_val)
                ax.plot(decay_dose, fit_curve, '-', linewidth=2.5, color='black', 
                       alpha=0.8, zorder=2, label=f'Exp Fit (R²={decay_r2:.3f})')
        
        # Customize axes
        ax.set_xlabel('Dose from Onset (MGy)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Absorbance at 412 nm', fontsize=11, fontweight='bold')
        
        # Add panel label
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, 
               fontsize=14, fontweight='bold', va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Update legend with transmission info
        legend_label = f'Data ({transmission_pct}% TX)'
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Add kinetics information as text
        kinetics_text = f'Dose Rate: {dose_data.get("dose_rate_MGy_s", 0):.4f} MGy/s\n'
        kinetics_text += f'Dose @ 10s: {dose_MGy[-1]:.3f} MGy'
        if sig_params:
            kinetics_text += f'\nIC50: {sig_params.get("IC50", decay_k):.3f} MGy'
            kinetics_text += f'\nn (Hill): {sig_params.get("n", 1.0):.2f}'
        
        ax.text(0.98, 0.02, kinetics_text, transform=ax.transAxes,
               fontsize=9, va='bottom', ha='right',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
               family='monospace')
    
    # Main title
    fig.suptitle('Dose-Dependent Decay of 412 nm Signal (Best-Fit Onset)',
                fontsize=15, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to {output_path}")
    plt.close()
    
    return fig


if __name__ == "__main__":
    print("This script is designed to be imported into labeled_spec_analysis.ipynb")
    print("It requires the 'dose_analysis' dictionary to be available in the notebook.")
