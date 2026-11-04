"""Evidence-based E11C classification; reads completed static records only."""
from pathlib import Path
from fractions import Fraction as F
import json
from src.drilling_reproduction import read,write
from src.drilling_constraint_verification import decode,bareiss_determinant,q


def nonzero_witness(a,lo=0,hi=None):
    found=[]
    for i,row in enumerate(a):
        for j,x in enumerate(row[lo:hi],lo):
            lower,upper=q(x['lo']),q(x['hi'])
            if lower>0 or upper<0:
                found.append({'row_zero_based':i,'column_zero_based':j,
                    'lo':x['lo'],'hi':x['hi'],'absolute_separation_display':float(lower if lower>0 else -upper)})
    return max(found,key=lambda x:x['absolute_separation_display']) if found else None


def analyze(out):
    out=Path(out);cfg=read(out/'protocol.json');raw=read(out/'interval_256.json')
    exact=read(out/'exact_dyadic.json');summary=read(out/'summary.json')
    groups={}
    for name,(lo,hi) in cfg['operator_columns'].items():
        witness=nonzero_witness(raw['native_constraint_acceleration'],lo,hi)
        groups[name]={'hypothesis':'Unreacted native acceleration is tangent for this fixed operator block',
            'status':'REJECTED_FOR_DECLARED_MODEL' if witness else 'UNRESOLVED',
            'nonzero_witness':witness,'unit_note':'Coefficient columns are algebraic bases; normalized magnitudes are not actuator limits or physical excitations.'}
    states=[]
    for i,(spec,col) in enumerate(zip(cfg['snapshots'],cfg['snapshot_columns'])):
        row=summary['arithmetic']['interval_256']['snapshots'][i]
        witness=nonzero_witness(raw['native_constraint_acceleration'],col,col+1)
        reaction=nonzero_witness(raw['retained_reaction_acceleration'],col,col+1)
        states.append({**spec,'constraint_status':'REJECTED_FOR_DECLARED_MODEL' if witness else 'UNRESOLVED',
            'constraint_witness':witness,'retained_reaction_nonzero_witness':reaction,
            'max_constraint_translation_m_s2':max(abs(x) for x in row['native_constraint_acceleration'][::2]),
            'max_constraint_rotation_rad_s2':max(abs(x) for x in row['native_constraint_acceleration'][1::2]),
            'max_reaction_retained_translation_m_s2':max(abs(x) for x in row['reaction_effect'][::2]),
            'max_reaction_retained_rotation_rad_s2':max(abs(x) for x in row['reaction_effect'][1::2])})
    hypotheses={
        'supplied_unreacted_force_already_enforces_constraints':{
            'status':'REJECTED_FOR_DECLARED_MODEL' if all(x['constraint_witness'] for x in states) else 'UNRESOLVED',
            'reason':'Four validated counterexamples plus nonzero fixed operator blocks; this does not prove physical plant invalidity.'},
        'ideal_reactions_have_no_effect_on_retained_accelerations':{
            'status':'REJECTED_FOR_DECLARED_MODEL' if all(x['retained_reaction_nonzero_witness'] for x in states) else 'UNRESOLVED'},
        'ideal_bilateral_reaction_formulations_are_algebraically_consistent':{
            'status':'SUPPORTED_CONDITIONALLY','reason':'Exact 31-column identities, nonsingular full block system, independent rational checks and validated interval solves.'},
        'archived_affine_reference_is_exact_for_native_projection':{
            'status':'REJECTED_FOR_DECLARED_MODEL' if nonzero_witness(raw['native_K_Vs']) else 'UNRESOLVED'},
        'archived_affine_reference_is_exact_for_ideal_projection':{
            'status':'REJECTED_FOR_DECLARED_MODEL' if nonzero_witness(raw['ideal_K_Vs']) else 'UNRESOLVED'},
        'ideal_reactions_are_the_actual_drilling_contact_model':{
            'status':'UNRESOLVED_PHYSICAL_ASSUMPTIONS','reason':'Normal-contact, friction, rotational kinematics, mesh/reference and load-closure assumptions are not supplied.'},
        'physical_drilling_plant_is_invalid':{'status':'UNRESOLVED','reason':'Four numerical snapshots and conditional mechanics cannot establish actual plant invalidity.'}}
    ranks={}
    for name in ('M','K','M_H','K_H'):
        determinant=bareiss_determinant(decode(exact[name]))
        expected=len(exact[name]);assert determinant!=0 and exact['exact_ranks'][name]==expected
        ranks[name]={'exact_dyadic_rank':expected,'determinant_nonzero':True,
            'determinant_exact':[str(determinant.numerator),str(determinant.denominator)]}
    cr=decode(exact['rigid_translation_constraint'])
    minor=bareiss_determinant([cr[0],cr[2]]);assert minor!=0
    ranks['rigid_translation_constraint_columns']={'rank':2,'nonzero_minor':[str(minor.numerator),str(minor.denominator)]}
    write(out/'additional_exact_rank_verification.json',ranks)
    write(out/'hypotheses.json',hypotheses)
    result={'operator_block_tests':groups,'snapshot_results':states,'hypotheses':hypotheses,
        'structured_rigid_translation_stiffness_zero_enclosed':all(q(x['lo'])<=0<=q(x['hi']) for row in raw['rigid_translation_stiffness'] for x in row),
        'interpretation':'Global linear coefficient counterexamples concern the declared fixed numerical model. Four archived snapshots are not a physical validity or safety study.'}
    write(out/'scientific_analysis.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    value=analyze(args.output or root/read(root/'configs/experiment_11c.json')['output'])
    print(json.dumps({'hypotheses':value['hypotheses'],'snapshots':value['snapshot_results']},indent=2))
