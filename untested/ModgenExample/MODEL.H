#define TIME_TYPE_DOUBLE

//If following file is not found by compiler, check paths for include file and for libraries.
//For information on how to do that, please consult developer's guide.
//Si le fichier suivant n'est pas trouvé par le compilateur, vérifiez les chemins pour les fichiers inclus et les librairies.
//Pour savoir comment modifier ces chemins, veuillez consulter le guide du concepteur de Modgen.
#include "Modgen12.h"

#include "stdafx.h"
#include "common.h"
namespace mm {
	typedef class Person* Person_ptr;

	inline void initialize_attributes() {}
	inline void enter_simulation() {}
	inline void exit_simulation() {}

}
#include "custom_early.h"
#include "actors.h"
#include "tabinit.h"
#include "custom.h"
