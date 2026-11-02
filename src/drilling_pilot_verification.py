"""E11A full saved-history verification and report; never advances a simulator."""
from pathlib import Path
import csv
import gc
import json
import shutil
import time
import numpy as np
from scipy.io import loadmat
from src.drilling_reproduction import read,write,sha,cases,A,B
from src.drilling_numerical_audit import external,dense,discrepancy,STATE_SCALE


def verify(root):
    root=Path(root).resolve();out=root/'results/drilling_reproduction/native_pilot'
    summary=read(out/'summary.json');extra=[];cost=0.;native_seconds=0.;capture_seconds=0.
    sample_total=0;before=read(out/'preservation_before.json')['sha256']
    for name,h in read(out/'source_hashes.json').items():assert sha(root/name)==h,name
    authority=read(out/'authority_preservation_before.json')
    for name,h in authority.items():assert sha(name)==h,name
    for name,h in before.items():assert sha(root/name)==h,name
    for case in summary['cases']:
        original=case['runs'][0];assert case['repeatability']['passed']
        nat=read(out/case['case_id']/'independent_native_safety.json')
        margins=list(original['safety']['minimum_margins'].values())
        assert nat['violating_states']==original['safety']['violation_state_count']
        assert nat['violations_by_face']==list(original['safety']['violations_by_face'].values())
        assert (nat['first_crossing_step'] or None)==original['safety']['first_crossing_step']
        matlab_safety_error=discrepancy(nat['minimum_margins'],margins)
        legacy={};first_crossings={}
        for phase in ['original','repeat']:
            folder=out/case['case_id']/phase;execution=read(folder/'execution.json')
            assert execution['process_returncode']==0 and not execution['timeout']
            assert execution['native_status']['native_entrypoint_completed']
            for name,h in execution['artifacts_sha256'].items():assert sha(folder/name)==h,name
            request=read(folder/'request.json')
            for name,h in request['copies_sha256'].items():assert sha(Path(request['execution_directory'])/name)==h,name
            audit=read(folder/'audit.json');assert audit['numerical_checks_passed']
            a=loadmat(folder/'native.mat');E=np.asarray(a['E1']);n=E.shape[1]
            assert n==200001 and not np.any(np.imag(E)) and np.isfinite(E).all()
            if case['case_id']=='P0':expected=np.fromstring(A+',-0.00045363,-1.7321',sep=',')
            elif case['case_id']=='P3':expected=np.r_[np.zeros(18),.00056292,4.5930]
            elif case['case_id']=='P5':expected=np.zeros(20)
            else:
                suffix={'P1':',0.00018,-3.6136','P2':',-0.00045363,-1.7321','P4':',0.0029012,2.1651'}[case['case_id']]
                expected=np.fromstring(B+suffix,sep=',')
            assert np.array_equal(E[:,0],expected)
            bits=np.ascontiguousarray(np.real(E[:,0]),dtype=np.float64).view(np.uint64)
            write(folder/'initial_state_identity.json',{'native_initial_error':np.real(E[:,0]).tolist(),
                'binary64_hex':[f'{int(x):016x}' for x in bits],'matches_approved_case':True})
            # Native F1_global stores computed loads at columns 2:n; first column is a placeholder.
            nominal=external(a,a['force_d'],a['torque_d'],a['WOB_d'],a['TOB_d'])
            applied=np.column_stack([nominal[:,0],np.asarray(a['F1_global'])[:,1:-1]])
            reconstructed=np.linalg.solve(dense(a['M1_global']),applied)[::2]-np.asarray(a['F_M_d'])
            reconstructed[:,0]=0 # Native initial error-load vector is explicitly zero.
            prediction=E[:,:-1]+1e-4*(np.asarray(a['Al1'])@E[:,:-1]+
                np.vstack([np.zeros_like(reconstructed),reconstructed]))
            full_step=discrepancy(prediction,E[:,1:],STATE_SCALE[:,None]);assert full_step['maximum_scaled']<1e-8
            np.savez_compressed(folder/'applied_vector_histories.npz',
                applied_external_force_vectors=applied,reconstructed_error_load_vectors=reconstructed,
                initial_error_load_vector=np.zeros(10))
            write(folder/'full_transition_audit.json',{'passed':True,'transitions':n-1,
                'discrepancy':full_step,'no_sequential_simulation':True,
                'reconstructed_error_load_vectors_are_independent_reconstruction_not_native_bitwise_claim':True,
                'initial_nominal_and_unused_terminal_semantics_preserved':True})
            if phase=='original':
                with np.load(folder/'derived_histories.npz') as derived:
                    m=derived['four_raw_margins']
                    legacy={'axial_minimum_excluding_initial':float(np.real(m[2:,1:]).min()),
                            'axial_minimum_including_initial':float(np.real(m[2:]).min())}
                    for i,key in enumerate(['omega_lower','omega_upper','axial_lower','axial_upper']):
                        indices=np.flatnonzero(np.real(m[i])<=0)
                        first_crossings[key]=None if not len(indices) else {'step':int(indices[0]),'seconds':float(indices[0]*1e-4)}
            cost+=execution['process_seconds'];native_seconds+=execution['native_status']['wall_seconds']
            capture_seconds+=read(folder/'capture.json')['core_seconds_including_save'];sample_total+=n
            extra.append({'case_id':case['case_id'],'phase':phase,'all_transition_audit':full_step,
                'native_vs_Python_margin_minima':matlab_safety_error,'first_crossings_by_face':first_crossings,
                'initial_sample_comparison':legacy})
            del a,E,applied,reconstructed,prediction;gc.collect()
    checks={'protocol_completed':True,'native_and_independent_numerical_verification_passed':True,
        'all_cases_safe':False,'exact_physical_reference_equilibrium_established':False,
        'bounded_actuator_feasibility':'BLOCKED; outside E11A scope',
        'technical_tests':read(out/'technical_tests.json'),'prior_project_files_preserved':len(before),
        'authority_files_preserved':len(authority),'source_copies_unchanged':True,
        'case_count':6,'native_executions':12,'integration_transitions':2400000,
        'raw_states_verified':sample_total,'native_numeric_field_repeat_comparisons':510,
        'process_wall_seconds':cost,'native_wrapper_seconds_including_save_and_plot':native_seconds,
        'core_capture_seconds_including_save':capture_seconds,
        'full_transition_checks':extra,'no_controller_change_or_refinement':True}
    assert checks['technical_tests']['passed']
    write(out/'verification.json',checks)
    additional=['src/matlab/e11a_reference_checks.m','src/drilling_pilot_verification.py']
    write(out/'additional_audit_sources.json',{name:sha(root/name) for name in additional})
    native=read(out/'native_reference_residuals.json')
    rows=[]
    for c in summary['cases']:
        r=c['runs'][0];m=r['safety']['minimum_margins'];inp=r['inputs']
        row={'case_id':c['case_id'],**m,'violating_states':r['safety']['violation_state_count'],
             'first_crossing_seconds':r['safety']['first_crossing_seconds'],'lookup_failures':r['lookup']['failures'],
             'peak_abs_force_N':inp['absolute_peak_force_N'],'peak_abs_torque_Nm':inp['absolute_peak_torque_Nm']}
        rows.append(row)
    with (out/'case_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    matrix_max=max(v['norm_relative'] for c in summary['cases'] for v in c['runs'][0]['matrix_reassembly'].values())
    step_abs=max(r['all_transition_audit']['maximum_absolute'] for r in extra)
    step_scaled=max(r['all_transition_audit']['maximum_scaled'] for r in extra)
    correction_abs=max(c['runs'][0]['lookup']['correction_discrepancy']['maximum_absolute'] for c in summary['cases'])
    correction_relative=max(c['runs'][0]['lookup']['correction_discrepancy']['norm_relative'] for c in summary['cases'])
    text=['# Experiment 11A — Authoritative ideal-input native reproduction pilot','',
      'Numerical reproduction completed. This is not a bounded-input safety study, controller redesign or RL experiment. The physical actuator contract remains blocked.','',
      '## Frozen execution and provenance','',
      'P0 uses the top-level script byte-for-byte. P1–P4 use the exact Chart_Plot initial errors and native IC_Test kernel. P5 is the newly approved zero-error input, not an archived case. The standalone Chart script is not substituted for P3. The ZIP SHA-256 is b86ce2a62fe26871f40eb674377b1b17ddedac84ef422019c47e59f1787be1b5. Source/data identities, exact expressions and executed initial binary64 values are retained.','',
      'MATLAB 24.2.0.2863752 (R2024b Update 5), MACA64, single computational thread, seed 111001. Base MATLAB was available. LMI synthesis functions were absent but are not a dependency of these native stored-table runs. Sandboxed startup failed before MATLAB output; authorized native execution succeeded.','',
      'Twelve separate processes used isolated temporary directories and read-only, byte-identical native files. IC_Test workspaces were saved using a false conditional breakpoint immediately before line 547, before display clipping; a fixture verified caller access and nonbreaking behavior. P0 was captured after native script completion. No authoritative source was edited.','',
      '## Six-case results','',
      '| Case | min omega lower (rad/s) | min omega upper (rad/s) | min axial lower (m/s) | min axial upper (m/s) | Violating states | First crossing (s) | Failed lookups |',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        text.append(f"| {r['case_id']} | {r['omega_lower_rad_s']:.12g} | {r['omega_upper_rad_s']:.12g} | {r['axial_lower_m_s']:.12g} | {r['axial_upper_m_s']:.12g} | {r['violating_states']} | {r['first_crossing_seconds'] if r['first_crossing_seconds'] is not None else 'none'} | {r['lookup_failures']} |")
    text += ['',
      'All 200,001 raw error states were checked in every execution. P0/P1/P2/P4/P5 remained strictly inside the sampled safety domain. P3 had 401 lower-rotational violations starting at 0.1323 s and 358 lower-axial violations starting at 0.1347 s. Its 411 lookup failures began at 0.1318 s; 401 attempts had at least one complex index. The native stale-gain/transformed-memory fallback was preserved and verified. No first lookup failed, so that failure path was not dynamically exercised or repaired.','',
      'All physical error/input/load histories were finite and real. P3 has complex transformed lookup indices, not complex physical trajectories. There were no native execution failures, timeouts or incomplete horizons. Failed or truncated histories are explicitly ineligible for safe classification.','',
      'P1/P4 have smaller axial margins than the historical context because the initial sample is now included. Excluding it gives 2.01414884471e-5 / 5.98700570543e-5 m/s, reproducing the earlier approximate values; including it gives 3.04250719781e-6 / 4.22019883481e-5. Earlier records were not changed.','',
      '## Reference consistency and equilibrium','',
      'P5 retained exactly zero error throughout and constant nominal inputs. This verifies the implemented nominal error map, not exact physical reference consistency. Native MATLAB gives:',
      f"- max |r0| = {max(abs(x) for x in native['r0']):.15g}; the largest component is angular acceleration in rad/s^2.",
      f"- max |K_M Vs| = {max(abs(x) for x in native['KM_Vs']):.15g}; the largest component is angular acceleration rate in rad/s^3.",
      f"- max |r0 - saved err| = {max(abs(x) for x in native['r0_minus_saved_err']):.15g}.",
      f"- max |r0 - 20 K_M Vs| = {max(abs(x) for x in native['physical_missing_term_at_20_seconds']):.15g} rad/s^2 in the dominant angular component.",
      '',
      'The conditions r0=0 and K_M Vs=0 are not satisfied exactly by the assembled matrices and stored reference. Independent FEM reassembly confirms the nonzero K_M Vs. The implemented error equation omits r0 - t K_M Vs. No compensation, recalculated IC, new controller or altered dynamics was introduced. The stiffness matrix condition estimate is approximately 4.03e17; small balance-residual differences must not be conflated with the much larger nonzero reference-stiffness term. These are failed physical-reference identities, separate from successful native reproduction.','',
      '## Verification and cost','',
      f"All {checks['technical_tests']['tests']} included tests passed, including five new E11A fixtures; unfinished IQL tests were excluded. Six native original/repeat pairs matched all 85 numerical workspace fields bitwise (510 field comparisons); MAT-file timestamps are excluded and gain-table identity is established by its source checksum. No cross-language bitwise equivalence is claimed.",
      f"Independent matrix reassembly worst norm-relative difference: {matrix_max:.6g}. All 2,400,000 recorded transitions were independently reconstructed without a sequential rollout; maximum absolute discrepancy {step_abs:.6g}, maximum state-scaled discrepancy {step_scaled:.6g}. Full controller correction reconstruction maximum absolute discrepancy {correction_abs:.6g}, worst norm-relative discrepancy {correction_relative:.6g}. Representative load/transform checks and independent MATLAB raw-safety checks passed. Absolute/scaled/per-variable diagnostics are retained in audit JSON files.",
      '',
      'State scaling for transition errors is 1 m/rad for displacement coordinates and 0.001 m/s / 10 rad/s for axial/angular velocity coordinates. The prespecified source checks require matrix norm-relative error <1e-10 and transition/load/transformation/correction scaled error <1e-8. These numerical tolerances never relax the strict safety boundaries.','',
      f"Twelve native runs: 2,400,000 integration transitions and 2,400,012 raw states. Process wall total {cost:.3f} s; native wrapper time including capture/plot {native_seconds:.3f} s; core capture time including MAT serialization {capture_seconds:.3f} s. Independent representative audit {summary['audit_seconds']:.3f} s; full-history verification and preservation overhead are separate. No broader integration refinement was run.",
      '',
      f"Preservation passed for {len(before)} pre-existing project files and {len(authority)} original authority files, including the ZIP, unpacked MATLAB/data files, Experiments 01–10, decision log and nine unchanged IQL files. All disposable authoritative copies remain byte-identical. No commit/push or prior experiment changes occurred.",
      '',
      '## Readiness and reproduction','',
      'Ready for scientific review of further numerical validation of the implemented ideal-input map. Not ready to assert exact physical-reference equivalence or bounded-input safety. Review the omitted reference term, initial physical-array/error mismatch and P3 fallback/domain exits before approving any next validation study. Optimization, alternatives, controller changes, training and integration refinement remain unperformed.','',
      'Generated evidence is under results/drilling_reproduction/native_pilot and is excluded from Git. native.mat retains original histories; derived_histories.npz and applied_vector_histories.npz label independent reconstructions, including the initial nominal loads and the 200,000 applied commands. The final computed command is retained separately and was not applied. Initial WOB/TOB array zeros are placeholders, not initial applied loads.','',
      'Reproduction requires the pinned native MATLAB release and exact external archive path/hash in configs/experiment_11a.json, plus the existing Python NumPy/SciPy environment. First run the documented MATLAB license check. Run experiments/11a_native_drilling.py --run only with appropriate native execution permission. Completed executions are verified/reused; interrupted artifacts cause a refusal to overwrite. --audit reads saved records only. e11a_reference_checks(root,output) performs static MATLAB safety/residual checks; python -m src.drilling_pilot_verification generates the final verification/report from those saved outputs. Use a disposable checkout/output location for any fresh reproduction.','']
    report='\n'.join(text);write(out/'report_metrics.json',{'cases':rows,'matrix_relative':matrix_max,
        'all_transition_absolute':step_abs,'all_transition_scaled':step_scaled,'correction_absolute':correction_abs,
        'correction_relative':correction_relative})
    (out/'scientific_report.md').write_text(report)
    doc=root/'docs/experiment_11a_report.md'
    if doc.exists():assert doc.read_text()==report,'Refusing to overwrite a differing report'
    else:doc.write_text(report)
    files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='completion.json'}
    write(out/'completion.json',{'protocol_completed':True,'numerical_checks_passed':True,
        'all_cases_safe':False,'physical_reference_identities_exact':False,
        'artifact_sha256':files,'report_sha256':sha(doc)})
    print(json.dumps({k:v for k,v in checks.items() if k!='full_transition_checks'},indent=2))

if __name__=='__main__':verify(Path(__file__).resolve().parents[1])
