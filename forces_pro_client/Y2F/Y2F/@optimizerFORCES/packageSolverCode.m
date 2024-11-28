function success = packageSolverCode(self)
%PACKAGESOLVERCODE Packages the code generated by
%GENERATECINTERFACECODE and GENERATEMEXINTERFACECODE.
%
% This file is part of the y2f project: http://github.com/embotech/y2f, 
% a project maintained by embotech under the MIT open-source license.
%
% (c) Gian Ulli and embotech AG, Zurich, Switzerland, 2013-2023.

success = 0;

solverName = self.default_codeoptions.name;
cName = [solverName '/interface/' solverName];
    
% move the (necessary) files of all solvers to the new directory and delete
% the folders of the "internal" solvers
for i=1:self.numSolvers
    % include
    dir2move = sprintf('%s/include',self.codeoptions{i}.name);
    if( exist(dir2move,'dir') )
        copyfile(dir2move, sprintf('%s/include',solverName), 'f');
    end
    % lib
    dir2move = sprintf('%s/lib',self.codeoptions{i}.name);
    if( exist(dir2move,'dir') )
        copyfile(dir2move, sprintf('%s/lib',solverName), 'f');
    end
    % obj
    dir2move = sprintf('%s/obj',self.codeoptions{i}.name);
    if( exist(dir2move,'dir') )
        copyfile(dir2move, sprintf('%s/obj',solverName), 'f');
    end
    % src
    dir2move = sprintf('%s/src',self.codeoptions{i}.name);
    if exist(dir2move,'dir')
        copyfile(dir2move, sprintf('%s/src',solverName), 'f');
    end
    % obj_target
    dir2move = sprintf('%s/obj_target',self.codeoptions{i}.name);
    if exist(dir2move,'dir')
        copyfile(dir2move, sprintf('%s/obj_target',solverName), 'f');
    end
    % lib_target
    dir2move = sprintf('%s/lib_target',self.codeoptions{i}.name);
    if exist(dir2move,'dir')
        copyfile(dir2move, sprintf('%s/lib_target',solverName), 'f');
    end
    % src_target
    dir2move = sprintf('%s/src_target',self.codeoptions{i}.name);
    if exist(dir2move,'dir')
        copyfile(dir2move, sprintf('%s/src_target',solverName), 'f');
    end
    
    % Delete files
    rmdir(self.codeoptions{i}.name, 's');
    delete([self.codeoptions{i}.name '*']);
end

success = 1;

end