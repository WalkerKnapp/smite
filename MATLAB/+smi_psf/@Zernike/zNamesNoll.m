function names = zNamesNoll(ll)
%zNamesNoll Classical names for Zernike indices ll (Noll ordering).

% Created by
%    Michael Wester, 2017, Lidkelab.

   names = arrayfun(@zNameNoll, ll, 'UniformOutput', false);

end

function name = zNameNoll(l)
% Classical name for Zernike index l (Noll ordering).

   % Noll ordering aberration names (1-based).  See
   % http://www.telescope-optics.net/zernike_expansion_schemes.htm
   Nnames = {'Piston',                         ...
             'Tilt Horizontal',                ...
             'Tilt Vertical',                  ...
             'Defocus',                        ...
             'Primary Astigmatism Oblique',    ...
             'Primary Astigmatism Vertical',   ...
             'Primary Coma Vertical',          ...
             'Primary Coma Horizontal',        ...
             'Trefoil Vertical',               ...
             'Trefoil Oblique',                ...
             'Primary Spherical',              ...
             'Secondary Astigmatism Vertical', ...
             'Secondary Astigmatism Oblique',  ...
             'Quadrafoil Vertical',            ...
             'Quadrafoil Oblique',             ...
             'Secondary Coma Horizontal',      ...
             'Secondary Coma Vertical',        ...
             'Secondary Trefoil Oblique',      ...
             'Secondary Trefoil Vertical',     ...
             'Pentafoil Oblique',              ...
             'Pentafoil Vertical'};

   if l <= 0
      error('zNameNoll: Zernike polynomials not defined for l <= 0!');
   else
      if l > numel(Nnames)
         name = sprintf('Zernike_Noll %d', l);
      else
         name = Nnames{l};
      end
   end

end
