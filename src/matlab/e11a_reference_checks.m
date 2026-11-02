function e11a_reference_checks(root,output)
% Static checks only: no state integration, controller execution or design.
r=jsondecode(fileread(fullfile(output,'P0','original','request.json')));
addpath(r.execution_directory);
a=load(fullfile(output,'P0','original','native.mat'), ...
    'K_M','D_M','Vs','IC','F_M_d','err');
r0=a.F_M_d-a.D_M*a.Vs-a.K_M*a.IC;
r1=a.K_M*a.Vs;
residual=struct('r0',r0,'KM_Vs',r1,'saved_err',a.err, ...
    'r0_minus_saved_err',r0-a.err,'physical_missing_term_at_20_seconds',r0-20*r1, ...
    'no_compensation_applied',true,'static_checks_only',true);
save(fullfile(output,'native_reference_residuals.mat'),'r0','r1');
fid=fopen(fullfile(output,'native_reference_residuals.json'),'w');
fprintf(fid,'%s\n',jsonencode(residual));fclose(fid);
for j=0:5
    id=sprintf('P%d',j);
    b=load(fullfile(output,id,'original','native.mat'),'E1','Vs','angle');
    c=cos(b.angle(end));
    vb=(b.E1(end-1,:)+b.Vs(end-1))/c;
    wb=(b.E1(end,:)+b.Vs(end))/c;
    ew=b.E1(end,:)/c;
    lo=arrayfun(@K_vl,ew);up=arrayfun(@K_vu,ew);
    margins=[wb-5.23;15.7-wb;vb-(.001+lo);.001+up-vb];
    valid=all(isfinite(b.E1),1)&all(imag(b.E1)==0,1);
    failed=valid & any(real(margins)<=0,1);
    first=find(failed,1);
    if isempty(first), first_step=[];else,first_step=first-1;end
    s=struct('case_id',id,'raw_state_count',size(b.E1,2), ...
        'minimum_margins',min(real(margins),[],2), ...
        'violations_by_face',sum(valid & real(margins)<=0,2), ...
        'violating_states',sum(failed),'first_crossing_step',first_step, ...
        'invalid_physical_states',sum(~valid),'maximum_absolute_error',max(abs(b.E1),[],'all'), ...
        'static_checks_only',true);
    fid=fopen(fullfile(output,id,'independent_native_safety.json'),'w');
    fprintf(fid,'%s\n',jsonencode(s));fclose(fid);
end
fprintf('E11A_STATIC_REFERENCE_AND_RAW_SAFETY_CHECKS_COMPLETE\n');
end
