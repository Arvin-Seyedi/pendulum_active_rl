function e11a_run(request_file)
% Execute unchanged sources. The IC_Test hook returns false and never pauses.
r = jsondecode(fileread(request_file));
setenv('E11A_OUTPUT',r.output);
set(0,'DefaultFigureVisible','off');
rng(r.seed,'twister');
cd(r.execution_directory);
environment=struct('version',version,'computer',computer,'toolboxes',ver, ...
    'base_license',license('test','MATLAB'),'synthesis_setlmis_exists',exist('setlmis','file'), ...
    'synthesis_required',false,'single_computational_thread_requested',true);
fid=fopen(fullfile(r.output,'environment.json'),'w');
fprintf(fid,'%s\n',jsonencode(environment));fclose(fid);
setappdata(0,'E11A_clock',tic);
try
    if strcmp(r.case_id,'P0')
        evalin('base','Direct_Drilling_5_Node_lateral_constrainted_LPV_barrier;');
        evalin('base','e11a_capture();');
    else
        dbstop('in','IC_Test','at','547','if','e11a_capture()');
        evalin('base',['IC_Test(' r.initial_expression ');']);
        dbclear in IC_Test
    end
    assert(isfile(fullfile(r.output,'native.mat')),'E11A:MissingCapture','No native snapshot');
    status=struct('native_entrypoint_completed',true,'exception_identifier','', ...
        'exception_report','','wall_seconds',toc(getappdata(0,'E11A_clock')));
catch exception
    status=struct('native_entrypoint_completed',false, ...
        'exception_identifier',exception.identifier, ...
        'exception_report',getReport(exception,'extended','hyperlinks','off'), ...
        'wall_seconds',toc(getappdata(0,'E11A_clock')));
    fprintf(2,'%s\n',status.exception_report);
    % P0 keeps its script workspace on failure; a failed function may not.
    if strcmp(r.case_id,'P0') && ~isfile(fullfile(r.output,'native.mat'))
        try, evalin('base','e11a_capture();'); catch, end
    end
end
fid=fopen(fullfile(r.output,'native_status.json'),'w');
fprintf(fid,'%s\n',jsonencode(status));fclose(fid);
close all;
fprintf('E11A %s native_entrypoint_completed=%d seconds=%.3f\n',r.case_id, ...
    status.native_entrypoint_completed,status.wall_seconds);
end
