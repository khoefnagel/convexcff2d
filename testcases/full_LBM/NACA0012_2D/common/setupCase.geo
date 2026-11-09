Include "caseDef.geo";

xmin = -width/4;
xmax = 3*width/4;
ymin = -height/2;
ymax = height/2;

nAirfoil = cellsPerChord*2;

nInflow = cellsPerChord/5;
nOutflow = cellsPerChord/5;

nTopBottom = cellsPerChord/2;
betaTopBottom = 1;


Include "NACA0012.geo";

//Points outside
//+
Point(400) = {xmin, ymin, 0, 1.0};
//+
Point(401) = {xmin, ymax, 0, 1.0};
//+
Point(402) = {xmax, ymin, 0, 1.0};
//+
Point(403) = {xmax, ymax, 0, 1.0};

//+
Line(2) = {401, 400};
//+
Line(3) = {400, 402};
//+
Line(4) = {401, 403};
//+
Line(5) = {403, 402};


//+
Transfinite Curve {1} = nAirfoil Using Progression 1;
//+
Transfinite Curve {2} = nInflow Using Progression 1;
//+
Transfinite Curve {5} = nOutflow Using Progression 1;
//+
Transfinite Curve {4, 3} = nTopBottom Using Progression betaTopBottom;
//+
Curve Loop(1) = {3, -5, -4, 2};
//+
Curve Loop(2) = {1};
//+
Plane Surface(1) = {1, 2};

Extrude {0, 0, c} {
  Surface{1};
  Layers{1};
  Recombine;
}



//+
Physical Surface("Sides", 33) = {1, 32};
//+
Physical Surface("Inlet", 34) = {27};
//+
Physical Surface("Outlet", 35) = {19};
//+
Physical Surface("Top", 36) = {23};
//+
Physical Surface("Bottom", 37) = {15};
//+
Physical Surface("Airfoil", 38) = {31};
//+
Physical Volume("Fluid", 39) = {1};
