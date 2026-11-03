"""Summarize completed static evidence; no FEM reassembly or trajectories."""
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from src.drilling_reproduction import read,write,sha


def analyze(out):
    out=Path(out);a=dict(np.load(out/'frozen_binary64.npz'))
    exact=read(out/'exact_dyadic.json');h=read(out/'interval_256.json')
    allpoints=[r for group in read(out/'pointwise_equations.json').values() for r in group]
    fraction=lambda pair:Q(int(pair[0]),int(pair[1]))
    r0lo,r0hi=[fraction(h['r0'][3][k]) for k in ('lo','hi')]
    r1lo,r1hi=[fraction(h['KM_Vs'][3][k]) for k in ('lo','hi')]
    missing=[r0lo-20*r1hi,r0hi-20*r1lo]
    coords=read(out/'coordinates_and_mechanics.json');c=np.array(coords['C']);s=np.array(coords['S']);t=np.array(coords['T'])
    velocity=t@a['Vs'][:,0]
    element_bit=a['R1'][:,:,-1]@velocity[-8:]
    node_bit=a['Vs'][-2:,0]/np.cos(a['angle'].reshape(-1)[-1])
    nominal=read(out/'nominal_balance.json')
    summary={'exact_archived_r0_max':max(abs(x) for x in exact['r0_display']),
      'exact_archived_KM_Vs_max':max(abs(x) for x in exact['KM_Vs_display']),
      'high_precision_r0_max':max(abs(x['midpoint_display']) for x in h['r0']),
      'high_precision_KM_Vs_max':max(abs(x['midpoint_display']) for x in h['KM_Vs']),
      'missing_acceleration_component4_at_20_seconds_exact_interval':[[str(x.numerator),str(x.denominator)] for x in missing],
      'missing_acceleration_component4_at_20_seconds_display':[float(x) for x in missing],
      'SM_inverse_CT_norm':float(np.linalg.norm(s@np.linalg.solve(a['M1_global'],c.T))),
      'nominal_bit_nodal_frame':[float(x) for x in node_bit],
      'nominal_bit_terminal_element_frame':[float(element_bit[4]),float(element_bit[6])],
      'element_nodal_frame_difference':[float(element_bit[4]-node_bit[0]),float(element_bit[6]-node_bit[1])],
      'pointwise_max_absolute_identity_error':max(r['maximum_absolute_identity_error'] for r in allpoints),
      'pointwise_max_scaled_identity_error':max(r['maximum_scaled_identity_error'] for r in allpoints),
      'pointwise_max_error_over_rounding_bound':max(r['maximum_error_over_rounding_bound'] for r in allpoints),
      'all_recorded_X_U_origin_errors_max':max(r['X_minus_U_minus_origin_max'] for r in allpoints),
      'post_initial_position_reconstruction_max':max(abs(x) for r in allpoints if r['step']>0 for x in r['native_position_minus_error_reconstruction']),
      'nominal_balance_max_identity_discrepancy':max(abs(x) for row in nominal['times'] for x in row['identity_discrepancy']),
      'interval_widths':{str(p):{key:max(x['width_display'] for x in read(out/f'interval_{p}.json')[key]) for key in ('r0','KM_Vs')} for p in (128,256)},
      'all_interval_residual_components_exclude_zero':all(x['excludes_zero'] for p in (128,256) for key in ('r0','KM_Vs') for x in read(out/f'interval_{p}.json')[key]),
      'scope':'Matched-state equation discrepancies, not trajectory deviations. All printed floats approximate exact rational/interval evidence.'}
    write(out/'analysis.json',summary)
    statuses={'source_provenance':{'status':'PASS','evidence':'ZIP, source/MAT hashes, frozen binary64 operands, archived original/repeat records'},
      'matrix_and_coordinate_reconstruction':{'status':'PASS','qualification':'Executed rotations/selection/origins reconstruct; literal printed rotation sign fails orthogonality and is retained as a documented source discrepancy.'},
      'algebraic_error_transformation':{'status':'PASS','qualification':'Correct derived equation includes r0-t K_M Vs; the native equation omits it.'},
      'exact_consistency_identities':{'status':'FAIL','qualification':'Both identities nonzero for exact archived operands and both independent-assembly precision enclosures.'},
      'numerical_conditioning':{'status':'PASS','qualification':'Diagnostics completed; K_M is severely ill-conditioned, not certified well-conditioned. Nonzero K_M Vs persists beyond rounding/assembly bounds.'},
      'physical_constrained_reduction_justification':{'status':'UNRESOLVED','qualification':'Source geometry constraints alone do not justify SM^-1 projection; unspecified reaction/contact assumptions remain.'}}
    write(out/'audit_statuses.json',statuses)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    print(analyze(args.output or root/read(root/'configs/experiment_11b.json')['output']))
