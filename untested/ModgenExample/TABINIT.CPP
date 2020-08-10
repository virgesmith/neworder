#include "model.h"

namespace mm {


double	Tab0Expr0( XTable *prTable, int nIndex )
{
	return prTable->GetVal(0, nIndex);
}

double	Tab0Expr1( XTable *prTable, int nIndex )
{
	return prTable->GetVal(3, nIndex);
}

double	Tab0Expr2( XTable *prTable, int nIndex )
{
	return prTable->GetVal(2, nIndex);
}

double	Tab0Expr3( XTable *prTable, int nIndex )
{
	return Division( prTable->GetVal(1, nIndex), prTable->GetVal(0, nIndex) );
}

#pragma optimize( "", off )
void InitTables()
{
	AddTable( 0,
		Tab0Expr0,
		Tab0Expr1,
		Tab0Expr2,
		Tab0Expr3 );
}

#pragma optimize( "", on )


int nTesTPaths = CheckPathsForLibraries_Modgen12();

} // namespace 
