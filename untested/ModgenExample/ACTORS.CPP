#include "model.h"

namespace mm {

#if defined(MG_CHKSUM)
	bool ::gbCheckSumActivated = true;
#else
	bool ::gbCheckSumActivated = false;
#endif

#pragma warning(disable : 4723)

ModelApp gApp( Simulation, InitParameters, DeleteParameters,
	ProcessModelGeneratedParameters, InitActors, DeleteActors,
	InitTables, PreSimulation, PostSimulation, UserTables,
	VerifyParameters, ValidateParameters, InitActorSets, DeleteActorSets );

void AFXAPI AfxAbort()
{
	TRACE0("AfxAbort called\n");
	DebugBreak();
#ifdef _WINDOWS
	AfxWinTerm();
#endif
	abort();
}

Thread EventQueue *gpoEventQueue;

Thread double ::gdThreadCurrentTime = 0;

Thread double ::gdThreadEvents = 0;

Thread double ::gdEventsForTabulation = 0;

Parameters	*gprParam;

BOOL VerifyParameters( CString *pszError)
{
	BOOL	bResult = TRUE;

	(*pszError).Empty();
	if ( !VerifyParameter( pszError, _T("MortalityHazard") ) ) bResult = FALSE;
	return bResult;
}

void InitParameters()
{
	gprParam = new Parameters();
}

void DeleteParameters()
{
	delete gprParam;
}

void SetCaseWeight( double dCaseWeight, double dCaseSubsampleWeight )
{
	gpoEventQueue->SetCaseWeight( dCaseWeight, dCaseSubsampleWeight ); 
}

// partitions

Parameters::Parameters()
{
	Initializer( &MortalityHazard, _T("MortalityHazard") );
}

void ProcessModelGeneratedParameters()
{
}

void InitActors()
{
	gpoEventQueue = new EventQueue();
	gpoEmptyPerson = NULL;
}

void DeleteActors()
{
	delete gpoEventQueue;
	gctrlSection.Lock();
	gctrlSection.Unlock();



	Person *poPerson = gpoEmptyPerson;
	while ( poPerson != NULL ) {
		Person *poPersonTemp = (Person *) poPerson->poNext;
		::delete poPerson;
		poPerson = poPersonTemp;
	}
}

// sizeof actors
int ::ganSizeOfActors[1] = {
	sizeof(Person) + sizeof(PersonDerivedStates) + sizeof(PersonTable),
};

void *Person::operator new( size_t count )
{
	Person *poActor;
	if ( gpoEmptyPerson != NULL ) {
		poActor = gpoEmptyPerson;
		gpoEmptyPerson = (Person *) gpoEmptyPerson->poNext;
	}
	else {
		poActor = ::new Person( TRUE ) ;
	}
	poActor->__finished = false;
	return poActor;
}

void Person::InitActor()
{
	poDerivedStates = new PersonDerivedStates(); 
	poDerivedStates->poParent = this;
	poStateFunctions = new PersonStateFunctions();
	poStateFunctions->poParent = this;
	poStateFunctions->poDerivedStates = poDerivedStates;
	poDerivedStates->poStateFunctions = poStateFunctions;
	bUpdates = FALSE;
	nActorNumber = 0;
	InitializeStates();
}

void Person::DeleteActor()
{
	delete poDerivedStates;
	delete poStateFunctions;
}

#pragma optimize( "", off )

void Person::InitializeStates()
{
	// initialization of states
	actor_id = (long) 0;
	actor_subsample_weight = (double) 0;
	actor_weight = (double) 0;
	age = (TIME) 0;
#line 35 "C:/Users/sabourin/Desktop/Chapter 1/ModgenExample/PersonCore.mpp"
	alive = TRUE != 0;
#line 145 "ACTORS.CPP"
	case_id = (long) 0;
	case_seed = (double) 0;
	time = (TIME) gdThreadCurrentTime;
	alive_value_in = FALSE;
	poDerivedStates->duration_ = (TIME) 0;
	poDerivedStates->duration__time = (TIME) gdThreadCurrentTime;
	duration__max_value_out = (TIME) 0;
	duration__min_value_out = (TIME) 0;
	table_filter_DurationOfLife = FALSE;
	__finished = FALSE;
	__time = (TIME) gdThreadCurrentTime;
	__events = 0;
}

#pragma optimize( "", on )

void Person::UpdateLinkedStates()
{
}

void Person::StartSpecial1()
{
	nSubSample = GetSubSample();
	InitializeStates();
	case_seed = GetCaseSeed();
	case_id = GetCaseID();
	lActorId = actor_id = GetObjectID();
	GetCaseWeight( &actor_weight, &actor_subsample_weight );

	// initialization of events
	poMortalityEvent = gpoEventQueue->NewEvent( this, 0, 0 );
	//Initalize derived states
#line 114 "C:/Users/sabourin/Desktop/Chapter 1/ModgenExample/PersonCore.mpp"
	poStateFunctions->Set_alive_value_in( logical( alive ) );
#line 180 "ACTORS.CPP"
#line 114 "C:/Users/sabourin/Desktop/Chapter 1/ModgenExample/PersonCore.mpp"
	poStateFunctions->Set_duration__max_value_out( TIME( poDerivedStates->duration_ ) );
#line 183 "ACTORS.CPP"
#line 114 "C:/Users/sabourin/Desktop/Chapter 1/ModgenExample/PersonCore.mpp"
	poStateFunctions->Set_duration__min_value_out( TIME( poDerivedStates->duration_ ) );
#line 186 "ACTORS.CPP"
	CountActor(0);
}

#pragma optimize( "", off )

void Person::StartSpecial2()
{
	// initialization of expression and history states
	poStateFunctions->Set_time(TIME( CoarsenMantissa( time ) ), true);
	poStateFunctions->Set_age(TIME( CoarsenMantissa( age ) ), true);
	gdThreadCurrentTime = time;
	__time = time;
	poDerivedStates->Set_duration_( (TIME) 0 );
	poDerivedStates->duration__time = time;
#line 108 "C:/Users/sabourin/Desktop/Chapter 1/ModgenExample/PersonCore.mpp"
	poStateFunctions->Set_table_filter_DurationOfLife( logical( 1 ) );
#line 203 "ACTORS.CPP"
	UpdateLinkedStates();

	// initialization of cross-tabulation updates
	tableinfo.tab0events = -1;
	tableinfo.tab0open0 = (real) ( alive_value_in );
	tableinfo.tab0open1 = (real) ( poDerivedStates->duration_ );
	tableinfo.tab0open2 = (real) ( duration__max_value_out );
	tableinfo.tab0open3 = (real) ( duration__min_value_out );
	poPrev = NULL;
	poNext = gpoEventQueue->poHeadActor;
	if ( gpoEventQueue->poHeadActor != NULL ) {
		gpoEventQueue->poHeadActor->poPrev = this;
	}
	gpoEventQueue->poHeadActor = this;
	bUpdates = TRUE;
}

void Person::FinishSpecial()
{
	CountFinishedActor(0);
	bUpdates = FALSE;
	gpoEventQueue->DestroyEvent( poMortalityEvent );

	// close all updates
	if ( gplTableIndexes[0] >= 0 && ( tableinfo.tab0dim_filter || table_filter_DurationOfLife ) ) {
		if ( tableinfo.tab0events != -1 ) {
			tableinfo.tab0open0 = (real) CloseUpdate( 0, 0, tableinfo.tab0open0, tableinfo.tab0close0, tableinfo.tab0dim_filter, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
			tableinfo.tab0open1 = (real) CloseUpdate( 0, 1, tableinfo.tab0open1, tableinfo.tab0close1, tableinfo.tab0dim_filter, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
			tableinfo.tab0open2 = (real) CloseUpdate( 0, 2, tableinfo.tab0open2, tableinfo.tab0close2, tableinfo.tab0dim_filter, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
			tableinfo.tab0open3 = (real) CloseUpdate( 0, 3, tableinfo.tab0open3, tableinfo.tab0close3, tableinfo.tab0dim_filter, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
		}
		tableinfo.tab0cell = (index) TableCellIndex( 0, 0 );
		if ( gpModelApp->m_nSubSamples == 1 ) {
			tableinfo.tab0total_cell = (index) tableinfo.tab0cell;
		}
		else {
			tableinfo.tab0total_cell = (index) TableTotalCellIndex( 0, 0 );
		}
		CloseUpdate( 0, 0, tableinfo.tab0open0, alive_value_in, table_filter_DurationOfLife, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
		CloseUpdate( 0, 1, tableinfo.tab0open1, poDerivedStates->Get_duration_(), table_filter_DurationOfLife, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
		CloseUpdate( 0, 2, tableinfo.tab0open2, poStateFunctions->Get_duration__max_value_out(), table_filter_DurationOfLife, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
		CloseUpdate( 0, 3, tableinfo.tab0open3, poStateFunctions->Get_duration__min_value_out(), table_filter_DurationOfLife, tableinfo.tab0cell, tableinfo.tab0total_cell, actor_weight, actor_subsample_weight );
	}

	// Remove actor from master list of active actors
	if ( poPrev != NULL )
	{
		poPrev->poNext = poNext;
	}
	else
	{
		if (gpoEventQueue->poHeadActor != NULL && gpoEventQueue->poHeadActor == this)
		{
			gpoEventQueue->poHeadActor = poNext;
		}
	}
	if ( poNext != NULL )
	{
		poNext->poPrev = poPrev;
	}
	poPrev = NULL;

	// Add actor to head of empty list of actors of this type
	poNext = gpoEmptyPerson;
	gpoEmptyPerson = this;
}

#pragma optimize( "", on )

void Person::WaitUntil( double target_time )
{
	if (!gpoEventQueue->m_bSpecialEvents)
	{
		//don't increase __events if special events because they are all considered one event for tabulation purposes
		__events = gdEventsForTabulation + 1;
	}
	if ( time < (TIME) target_time )
	{
		__time = (TIME) target_time;
		WaitSpecial( (TIME) target_time - time );
		poStateFunctions->Set_time( (TIME) target_time );
	}
}

void Person::WaitUntilThisActor( double target_time )
{
	WaitUntil(target_time);
}

void Person::WaitSpecial( TIME wait_time )
{

	// increment the age
		poStateFunctions->Set_age( age + wait_time );
}

double Person::EventTime( int nEventNum, int *pnEventInfo )
{
	double	event_time = double( TIME_INFINITE );

	gdRuntimeTime = (double) time;
	gnRuntimeActorId = actor_id;
	switch( nEventNum ) {
		case 0:
			gszRuntimeEventName = _T("Person.MortalityEvent(time)");
			event_time = CoarsenMantissa( (TIME) timeMortalityEvent() );
			TRCEVNT(case_seed, _T("Person"), actor_id, _T("timeMortalityEvent"), event_time);
			break;
		case -1:
		default:;
	}
	return (TIME) event_time;
}

void Person::Implement( int nEventNum, int nEventInfo )
{
	gdThreadEvents += 1;
	if (!gpoEventQueue->m_bSpecialEvents)
	{
		gdEventsForTabulation++;
	}
	gdRuntimeTime = (double) time;
	gnRuntimeActorId = actor_id;
	switch( nEventNum )
	{
		case 0:
			CHKSUM(time, 0);
			TRCEVNT(case_seed, _T("Person"), actor_id, _T("Person.MortalityEvent"), time);
			gszRuntimeEventName = _T("Person.MortalityEvent(implement)");
			MortalityEvent();
			break;
		case -1:
		default:;
	}
}

double Person::Set_actor_subsample_weight( double value )
{
	return poStateFunctions->Set_actor_subsample_weight( value);
}

double Person::Set_actor_weight( double value )
{
	return poStateFunctions->Set_actor_weight( value);
}

logical PersonStateFunctions::Set_alive( logical value )
{

	value = value != 0;
	if ( poParent->alive != value )
	{


		logical	OldValue;
		OldValue = poParent->alive;
		poParent->alive = value;

#line 114 "C:/Users/sabourin/Desktop/Chapter 1/ModgenExample/PersonCore.mpp"
		Set_alive_value_in( poParent->alive );
#line 364 "ACTORS.CPP"
	}
	return poParent->alive = value;
}

logical PersonStateFunctions::Set_alive_value_in( logical value )
{

	value = value != 0;
	if ( poParent->alive_value_in != value )
	{


		logical	OldValue;
		OldValue = poParent->alive_value_in;
		poParent->alive_value_in = value;

		if ( poParent->bUpdates ) {
			if ( gplTableIndexes[0] >= 0 && poParent->tableinfo.tab0events == poParent->__events ) {
				poParent->tableinfo.tab0close0 = (real) poParent->alive_value_in;
			}
		}
	}
	return poParent->alive_value_in = value;
}

TIME PersonStateFunctions::Set_duration__max_value_out( TIME value )
{

	if ( poParent->duration__max_value_out != value )
	{


		TIME	OldValue;
		OldValue = poParent->duration__max_value_out;
		poParent->duration__max_value_out = value;

		if ( poParent->bUpdates ) {
			if ( gplTableIndexes[0] >= 0 && poParent->tableinfo.tab0events == poParent->__events ) {
				poParent->tableinfo.tab0close2 = (real) poParent->duration__max_value_out;
			}
		}
	}
	return poParent->duration__max_value_out = value;
}

TIME	PersonStateFunctions::Get_duration__max_value_out(){
return poDerivedStates->Get_duration_();
};

TIME PersonStateFunctions::Set_duration__min_value_out( TIME value )
{

	if ( poParent->duration__min_value_out != value )
	{


		TIME	OldValue;
		OldValue = poParent->duration__min_value_out;
		poParent->duration__min_value_out = value;

		if ( poParent->bUpdates ) {
			if ( gplTableIndexes[0] >= 0 && poParent->tableinfo.tab0events == poParent->__events ) {
				poParent->tableinfo.tab0close3 = (real) poParent->duration__min_value_out;
			}
		}
	}
	return poParent->duration__min_value_out = value;
}

TIME	PersonStateFunctions::Get_duration__min_value_out(){
return poDerivedStates->Get_duration_();
};

logical PersonStateFunctions::Set_table_filter_DurationOfLife( logical value )
{

	value = value != 0;
	if ( poParent->table_filter_DurationOfLife != value )
	{


		logical	OldValue;
		OldValue = poParent->table_filter_DurationOfLife;
		poParent->table_filter_DurationOfLife = value;

		if ( poParent->bUpdates ) {
			if ( gplTableIndexes[0] >= 0 ) {
				if ( poParent->tableinfo.tab0events == -1 || poParent->tableinfo.tab0events < poParent->__events ) {
					if ( poParent->tableinfo.tab0events != -1 ) {
						poParent->tableinfo.tab0open0 = (real) poParent->CloseUpdate( 0, 0, poParent->tableinfo.tab0open0, poParent->tableinfo.tab0close0, poParent->tableinfo.tab0dim_filter, poParent->tableinfo.tab0cell, poParent->tableinfo.tab0total_cell, poParent->actor_weight, poParent->actor_subsample_weight );
						poParent->tableinfo.tab0open1 = (real) poParent->CloseUpdate( 0, 1, poParent->tableinfo.tab0open1, poParent->tableinfo.tab0close1, poParent->tableinfo.tab0dim_filter, poParent->tableinfo.tab0cell, poParent->tableinfo.tab0total_cell, poParent->actor_weight, poParent->actor_subsample_weight );
						poParent->tableinfo.tab0open2 = (real) poParent->CloseUpdate( 0, 2, poParent->tableinfo.tab0open2, poParent->tableinfo.tab0close2, poParent->tableinfo.tab0dim_filter, poParent->tableinfo.tab0cell, poParent->tableinfo.tab0total_cell, poParent->actor_weight, poParent->actor_subsample_weight );
						poParent->tableinfo.tab0open3 = (real) poParent->CloseUpdate( 0, 3, poParent->tableinfo.tab0open3, poParent->tableinfo.tab0close3, poParent->tableinfo.tab0dim_filter, poParent->tableinfo.tab0cell, poParent->tableinfo.tab0total_cell, poParent->actor_weight, poParent->actor_subsample_weight );
					}
					poParent->tableinfo.tab0events = poParent->__events;
					poParent->tableinfo.tab0close0 = (real) poParent->alive_value_in;
					poParent->tableinfo.tab0close1 = (real) poDerivedStates->Get_duration_();
					poParent->tableinfo.tab0close2 = (real) Get_duration__max_value_out();
					poParent->tableinfo.tab0close3 = (real) Get_duration__min_value_out();
					poParent->tableinfo.tab0cell = (index) poParent->TableCellIndex( 0, 0 );
					if ( gpModelApp->m_nSubSamples == 1 ) {
						poParent->tableinfo.tab0total_cell = (index) poParent->tableinfo.tab0cell;
					}
					else {
						poParent->tableinfo.tab0total_cell = (index) poParent->TableTotalCellIndex( 0, 0 );
					}
					poParent->tableinfo.tab0dim_filter = OldValue;
				}
			}
		}
	}
	return poParent->table_filter_DurationOfLife = value;
}

TIME PersonDerivedStates::Set_duration_( TIME value )
{

	if ( duration_ != value ) {

		TIME OldValue;
		OldValue = duration_;
		duration_ = value;
		poStateFunctions->Set_duration__max_value_out( duration_ );
		poStateFunctions->Set_duration__min_value_out( duration_ );
		if ( poParent->bUpdates ) {
			if ( gplTableIndexes[0] >= 0 ) {
				if ( poParent->tableinfo.tab0events == poParent->__events ) {
					poParent->tableinfo.tab0close1 = (real) duration_;
				}
			}
		}
	}
	return duration_ = value;
}
Thread Person	*gpoEmptyPerson = NULL;

void DeleteAllPersonActors()
{
}

void PreSimulation() {
}

void PostSimulation() {
}

void UserTables() {
}

BOOL ValidateParameters( SCENARIO_EVENT eEvent ) {
	BOOL bProceed = TRUE;

	 return bProceed;
}

void InitActorSets()
{

}

void	DeleteActorSets()
{

}

} // namespace 
