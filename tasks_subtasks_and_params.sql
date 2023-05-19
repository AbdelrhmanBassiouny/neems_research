SELECT task.dul_Task_o as task, taskt.o as task_type, task_ib.o as task_start, task_ie.o as task_end,
       subtask.dul_Task_o as subtask,subtaskt.o as subtask_type,
       subtask_ib.o as subtask_start, subtask_ie.o as subtask_end, subtask_hp.dul_Object_o as participant,
       participant_type.o as participant_type, parameter_type2.o as subtask_param,
       subtask_es.soma_ExecutionStateRegion_o as subtask_state, task_es.soma_ExecutionStateRegion_o as task_state,
       n.name as neem_name, n.description as neem_desc, n.actname as activity, ne.environment_values as environment,
       n._id as neem_id
From dul_hasConstituent as hc
INNER JOIN dul_executesTask as task
on hc.dul_Entity_s = task.dul_Action_s and hc.neem_id = task.neem_id
INNER JOIN dul_executesTask as subtask
on hc.dul_Entity_o = subtask.dul_Action_s and hc.neem_id = subtask.neem_id
INNER JOIN rdf_type as taskt
On task.dul_Task_o = taskt.s and taskt.o != 'owl:NamedIndividual' and task.neem_id = taskt.neem_id
INNER JOIN rdf_type as subtaskt
ON subtask.dul_Task_o = subtaskt.s and subtaskt.o != 'owl:NamedIndividual' and subtask.neem_id = subtaskt.neem_id
Left JOIN (Select hpara.dul_Concept_s, dc.dul_Entity_o, hpara.dul_Parameter_o, hpara.neem_id
        From dul_hasParameter as hpara
            INNER JOIN dul_classifies as dc
                ON dc.dul_Concept_s = hpara.dul_Parameter_o and dc.neem_id = hpara.neem_id
        ) as hpara2
On hpara2.dul_Concept_s = subtask.dul_Task_o and hpara2.neem_id = subtask.neem_id
Left JOIN rdf_type as  parameter_type2
ON hpara2.dul_Entity_o = parameter_type2.s and parameter_type2.o != 'owl:NamedIndividual' and hpara2.neem_id = parameter_type2.neem_id
Inner JOIN dul_hasTimeInterval task_ti on task.dul_Action_s = task_ti.dul_Event_s and task.neem_id = task_ti.neem_id
INNER JOIN dul_hasTimeInterval subtask_ti on subtask.dul_Action_s = subtask_ti.dul_Event_s and subtask.neem_id = subtask_ti.neem_id
Inner JOIN soma_hasIntervalBegin task_ib on task_ti.dul_TimeInterval_o = task_ib.dul_TimeInterval_s and task_ti.neem_id = task_ib.neem_id
Inner JOIN soma_hasIntervalBegin subtask_ib on subtask_ti.dul_TimeInterval_o = subtask_ib.dul_TimeInterval_s and subtask_ti.neem_id = subtask_ib.neem_id
Inner JOIN soma_hasIntervalEnd task_ie on task_ti.dul_TimeInterval_o = task_ie.dul_TimeInterval_s and task_ti.neem_id = task_ie.neem_id
Inner JOIN soma_hasIntervalEnd subtask_ie on subtask_ti.dul_TimeInterval_o = subtask_ie.dul_TimeInterval_s and subtask_ti.neem_id = subtask_ie.neem_id
Left JOIN dul_hasParticipant as subtask_hp
ON subtask_hp.dul_Event_s = subtask.dul_Action_s and subtask_hp.neem_id = subtask.neem_id
Left JOIN rdf_type as  participant_type
ON subtask_hp.dul_Object_o = participant_type.s and participant_type.o != 'owl:NamedIndividual'
       and participant_type.neem_id = subtask_hp.neem_id
Left JOIN soma_hasExecutionState task_es on task_es.dul_Action_s = hc.dul_Entity_s and task_es.neem_id = hc.neem_id
Left JOIN soma_hasExecutionState subtask_es on subtask_es.dul_Action_s = hc.dul_Entity_o and subtask_es.neem_id = hc.neem_id
# Left join neems n on n._id = hc.neem_id
Left Join (Select _id, n.name, n.description, n.ID, na.name as actname
           From neems as n
                    Left Join (Select na.ID, nai.neems_ID, na.name
                               From neems_activity as na
                                        Left JOIN neems_activity_index nai on na.ID = nai.neems_activity_ID) as na
                              on na.neems_ID = n.ID) as n on n._id = hc.neem_id
Left Join neems_environment_index ne on ne.neems_ID = n.ID
Order by subtask_ib.o;