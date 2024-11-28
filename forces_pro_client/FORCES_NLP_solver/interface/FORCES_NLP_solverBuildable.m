% FORCES_NLP_solver : A fast customized optimization solver.
% 
% Copyright (C) 2013-2023 EMBOTECH AG [info@embotech.com]. All rights reserved.
% 
% 
% This software is intended for simulation and testing purposes only. 
% Use of this software for any commercial purpose is prohibited.
% 
% This program is distributed in the hope that it will be useful.
% EMBOTECH makes NO WARRANTIES with respect to the use of the software 
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
% PARTICULAR PURPOSE. 
% 
% EMBOTECH shall not have any liability for any damage arising from the use
% of the software.
% 
% This Agreement shall exclusively be governed by and interpreted in 
% accordance with the laws of Switzerland, excluding its principles
% of conflict of laws. The Courts of Zurich-City shall have exclusive 
% jurisdiction in case of any dispute.
% 
classdef FORCES_NLP_solverBuildable < coder.ExternalDependency

    methods (Static)
        
        function name = getDescriptiveName(~)
            name = mfilename;
        end
        
        function b = isSupportedContext(context)
            b = context.isMatlabHostTarget();
        end
        
        function updateBuildInfo(buildInfo, cfg)
            buildablepath = fileparts(mfilename('fullpath'));
            [solverpath, foldername] = fileparts(buildablepath);
            [~, solvername] = fileparts(solverpath);
            % if the folder structure does not match to the interface folder, we assume it's the directory that contains the solver
            if(~strcmp(foldername, 'interface') || ~strcmp(solvername, 'FORCES_NLP_solver'))
                solverpath = fullfile(buildablepath, 'FORCES_NLP_solver');
            end
            solverInfo = struct();
            solverInfo.solvername = 'FORCES_NLP_solver';
            solverInfo.solverpath = solverpath;
            solverInfo.pythonClientFormat = true;
            solverInfo.useParallel = 0;
            solverInfo.isNLP = true;
            ForcesUpdateBuildInfo(buildInfo, cfg, solverInfo);
            postUpdateBuildInfoScript = [solverInfo.solvername, 'PostUpdateBuildInfo'];
            if exist(fullfile(buildablepath, [postUpdateBuildInfoScript, '.m']), 'file')
                postUpdateBuildInfo = str2func(postUpdateBuildInfoScript);
                postUpdateBuildInfo(buildInfo, cfg, solverInfo);
            end
        end
        
        function [output, exitflag, info] = forcesInitOutputsMatlab()
            infos_it = coder.nullcopy(zeros(1, 1));
            infos_res_eq = coder.nullcopy(zeros(1, 1));
            infos_rsnorm = coder.nullcopy(zeros(1, 1));
            infos_pobj = coder.nullcopy(zeros(1, 1));
            infos_solvetime = coder.nullcopy(zeros(1, 1));
            infos_fevalstime = coder.nullcopy(zeros(1, 1));
            infos_QPtime = coder.nullcopy(zeros(1, 1));
            infos_QPit = coder.nullcopy(zeros(1, 1));
            infos_QPexitflag = coder.nullcopy(zeros(1, 1));
            infos_solver_id = coder.nullcopy(zeros(8, 1));
            info = struct('it', infos_it,...
                          'res_eq', infos_res_eq,...
                          'rsnorm', infos_rsnorm,...
                          'pobj', infos_pobj,...
                          'solvetime', infos_solvetime,...
                          'fevalstime', infos_fevalstime,...
                          'QPtime', infos_QPtime,...
                          'QPit', infos_QPit,...
                          'QPexitflag', infos_QPexitflag,...
                          'solver_id', infos_solver_id);

            outputs_x01 = coder.nullcopy(zeros(8, 1));
            outputs_x02 = coder.nullcopy(zeros(8, 1));
            outputs_x03 = coder.nullcopy(zeros(8, 1));
            outputs_x04 = coder.nullcopy(zeros(8, 1));
            outputs_x05 = coder.nullcopy(zeros(8, 1));
            outputs_x06 = coder.nullcopy(zeros(8, 1));
            outputs_x07 = coder.nullcopy(zeros(8, 1));
            outputs_x08 = coder.nullcopy(zeros(8, 1));
            outputs_x09 = coder.nullcopy(zeros(8, 1));
            outputs_x10 = coder.nullcopy(zeros(8, 1));
            outputs_x11 = coder.nullcopy(zeros(8, 1));
            outputs_x12 = coder.nullcopy(zeros(8, 1));
            outputs_x13 = coder.nullcopy(zeros(8, 1));
            outputs_x14 = coder.nullcopy(zeros(8, 1));
            outputs_x15 = coder.nullcopy(zeros(8, 1));
            outputs_x16 = coder.nullcopy(zeros(8, 1));
            outputs_x17 = coder.nullcopy(zeros(8, 1));
            outputs_x18 = coder.nullcopy(zeros(8, 1));
            outputs_x19 = coder.nullcopy(zeros(8, 1));
            outputs_x20 = coder.nullcopy(zeros(8, 1));
            outputs_x21 = coder.nullcopy(zeros(8, 1));
            output = struct('x01', outputs_x01,...
                            'x02', outputs_x02,...
                            'x03', outputs_x03,...
                            'x04', outputs_x04,...
                            'x05', outputs_x05,...
                            'x06', outputs_x06,...
                            'x07', outputs_x07,...
                            'x08', outputs_x08,...
                            'x09', outputs_x09,...
                            'x10', outputs_x10,...
                            'x11', outputs_x11,...
                            'x12', outputs_x12,...
                            'x13', outputs_x13,...
                            'x14', outputs_x14,...
                            'x15', outputs_x15,...
                            'x16', outputs_x16,...
                            'x17', outputs_x17,...
                            'x18', outputs_x18,...
                            'x19', outputs_x19,...
                            'x20', outputs_x20,...
                            'x21', outputs_x21);
            
            exitflag = coder.nullcopy(0);
        end

        function [output,exitflag,info] = forcesCallWithParams(params)
            [output,exitflag,info] = FORCES_NLP_solverBuildable.forcesCall(params.xinit, params.x0, params.all_parameters, params.reinitialize);
        end

        function [output, exitflag, info] = forcesCall(xinit, x0, all_parameters, reinitialize)
            solvername = 'FORCES_NLP_solver';

            
            params = struct('xinit', double(xinit),...
                            'x0', double(x0),...
                            'all_parameters', double(all_parameters),...
                            'reinitialize', int32(reinitialize));

            [output_c, exitflag_c, info_c] = FORCES_NLP_solverBuildable.forcesInitOutputsC(); %#ok<ASGLU>
            
            headerName = [solvername '.h'];
            coder.cinclude(headerName);
            coder.cinclude([solvername '_memory.h']);
            coder.cinclude([solvername '_adtool2forces.h']);
            % define memory pointer
            memptr = coder.opaque([solvername '_mem *'], 'HeaderFile', headerName); %#ok<NASGU>
            memptr = coder.ceval([solvername '_internal_mem'], uint32(0));
            % define solver input information (params, file and casadi)
            coder.cstructname(params, [solvername '_params'], 'extern', 'HeaderFile', headerName);
            fp = coder.opaque('FILE *', 'NULL', 'HeaderFile', headerName);
            % need define extern int solvername_adtool2forces(solvername_float *x, solvername_float *y, solvername_float *l, solvername_float *p, solvername_float *f, solvername_float *nabla_f, solvername_float *c, solvername_float *nabla_c, solvername_float *h, solvername_float *nabla_h, solvername_float *hess, solver_int32_default stage, solver_int32_default iteration);
            casadi = coder.opaque([solvername '_extfunc'],['&' solvername '_adtool2forces'],'HeaderFile',headerName);
            % define solver output information (output, exitflag, info)
            coder.cstructname(output_c,[solvername '_output'], 'extern', 'HeaderFile', headerName);
            coder.cstructname(info_c,[solvername '_info'], 'extern', 'HeaderFile', headerName);
            exitflag_c = coder.ceval([solvername '_solve'], coder.rref(params), ...
                                      coder.wref(output_c), coder.wref(info_c), ... 
                                      memptr, fp, casadi);
            
            [output, exitflag, info] = FORCES_NLP_solverBuildable.forcesInitOutputsMatlab();

            info.it = cast(info_c.it, 'like', info.it);
            info.res_eq = cast(info_c.res_eq, 'like', info.res_eq);
            info.rsnorm = cast(info_c.rsnorm, 'like', info.rsnorm);
            info.pobj = cast(info_c.pobj, 'like', info.pobj);
            info.solvetime = cast(info_c.solvetime, 'like', info.solvetime);
            info.fevalstime = cast(info_c.fevalstime, 'like', info.fevalstime);
            info.QPtime = cast(info_c.QPtime, 'like', info.QPtime);
            info.QPit = cast(info_c.QPit, 'like', info.QPit);
            info.QPexitflag = cast(info_c.QPexitflag, 'like', info.QPexitflag);
            info.solver_id = cast(info_c.solver_id, 'like', info.solver_id);

            output.x01 = cast(output_c.x01, 'like', output.x01);
            output.x02 = cast(output_c.x02, 'like', output.x02);
            output.x03 = cast(output_c.x03, 'like', output.x03);
            output.x04 = cast(output_c.x04, 'like', output.x04);
            output.x05 = cast(output_c.x05, 'like', output.x05);
            output.x06 = cast(output_c.x06, 'like', output.x06);
            output.x07 = cast(output_c.x07, 'like', output.x07);
            output.x08 = cast(output_c.x08, 'like', output.x08);
            output.x09 = cast(output_c.x09, 'like', output.x09);
            output.x10 = cast(output_c.x10, 'like', output.x10);
            output.x11 = cast(output_c.x11, 'like', output.x11);
            output.x12 = cast(output_c.x12, 'like', output.x12);
            output.x13 = cast(output_c.x13, 'like', output.x13);
            output.x14 = cast(output_c.x14, 'like', output.x14);
            output.x15 = cast(output_c.x15, 'like', output.x15);
            output.x16 = cast(output_c.x16, 'like', output.x16);
            output.x17 = cast(output_c.x17, 'like', output.x17);
            output.x18 = cast(output_c.x18, 'like', output.x18);
            output.x19 = cast(output_c.x19, 'like', output.x19);
            output.x20 = cast(output_c.x20, 'like', output.x20);
            output.x21 = cast(output_c.x21, 'like', output.x21);
            
            exitflag = cast(exitflag_c, 'like', exitflag);
        end
    end

    methods (Static, Access = private)
        function [output, exitflag, info] = forcesInitOutputsC()
            infos_it = coder.nullcopy(int32(zeros(1, 1)));
            infos_res_eq = coder.nullcopy(double(zeros(1, 1)));
            infos_rsnorm = coder.nullcopy(double(zeros(1, 1)));
            infos_pobj = coder.nullcopy(double(zeros(1, 1)));
            infos_solvetime = coder.nullcopy(double(zeros(1, 1)));
            infos_fevalstime = coder.nullcopy(double(zeros(1, 1)));
            infos_QPtime = coder.nullcopy(double(zeros(1, 1)));
            infos_QPit = coder.nullcopy(int32(zeros(1, 1)));
            infos_QPexitflag = coder.nullcopy(int32(zeros(1, 1)));
            infos_solver_id = coder.nullcopy(int32(zeros(8, 1)));
            info = struct('it', infos_it,...
                          'res_eq', infos_res_eq,...
                          'rsnorm', infos_rsnorm,...
                          'pobj', infos_pobj,...
                          'solvetime', infos_solvetime,...
                          'fevalstime', infos_fevalstime,...
                          'QPtime', infos_QPtime,...
                          'QPit', infos_QPit,...
                          'QPexitflag', infos_QPexitflag,...
                          'solver_id', infos_solver_id);
                          
            outputs_x01 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x02 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x03 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x04 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x05 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x06 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x07 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x08 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x09 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x10 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x11 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x12 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x13 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x14 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x15 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x16 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x17 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x18 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x19 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x20 = coder.nullcopy(double(zeros(8, 1)));
            outputs_x21 = coder.nullcopy(double(zeros(8, 1)));
            output = struct('x01', outputs_x01,...
                            'x02', outputs_x02,...
                            'x03', outputs_x03,...
                            'x04', outputs_x04,...
                            'x05', outputs_x05,...
                            'x06', outputs_x06,...
                            'x07', outputs_x07,...
                            'x08', outputs_x08,...
                            'x09', outputs_x09,...
                            'x10', outputs_x10,...
                            'x11', outputs_x11,...
                            'x12', outputs_x12,...
                            'x13', outputs_x13,...
                            'x14', outputs_x14,...
                            'x15', outputs_x15,...
                            'x16', outputs_x16,...
                            'x17', outputs_x17,...
                            'x18', outputs_x18,...
                            'x19', outputs_x19,...
                            'x20', outputs_x20,...
                            'x21', outputs_x21);
            exitflag = coder.nullcopy(int32(0));
        end
    end

    
end
