SELECT task.dul_Task_o as task, taskt.o as task_type, task_ib.o as task_start, task_ie.o as task_end,
       task_es.soma_ExecutionStateRegion_o as task_state, t_sat.dul_Description_o as task_failure_type,
       task_hp.dul_Object_o as task_participant, participant_type.o as task_participant_type,
       #parameter_type.o as task_param,
       n.name as neem_name, n.description as neem_desc, n.actname as activity, ne.environment_values as environment,
       n._id as neem_id, n.created_by as created_by
From dul_hasConstituent as hc
INNER JOIN dul_executesTask as task
on hc.dul_Entity_s = task.dul_Action_s and hc.neem_id = task.neem_id
INNER JOIN rdf_type as taskt
On task.dul_Task_o = taskt.s and taskt.o != 'owl:NamedIndividual' and task.neem_id = taskt.neem_id and taskt.o not Regexp '^soma:Phy'
Inner JOIN dul_hasTimeInterval task_ti on task.dul_Action_s = task_ti.dul_Event_s and task.neem_id = task_ti.neem_id
Inner JOIN soma_hasIntervalBegin task_ib on task_ti.dul_TimeInterval_o = task_ib.dul_TimeInterval_s and task_ti.neem_id = task_ib.neem_id
Inner JOIN soma_hasIntervalEnd task_ie on task_ti.dul_TimeInterval_o = task_ie.dul_TimeInterval_s and task_ti.neem_id = task_ie.neem_id
Left JOIN soma_hasExecutionState task_es on task_es.dul_Action_s = hc.dul_Entity_s and task_es.neem_id = hc.neem_id
Left Join (Select _id, n.name, n.description, n.ID, na.name as actname, n.created_by
           From neems as n
                    Left Join (Select na.ID, nai.neems_ID, na.name
                               From neems_activity as na
                                        Left JOIN neems_activity_index nai on na.ID = nai.neems_activity_ID) as na
                              on na.neems_ID = n.ID) as n on n._id = hc.neem_id
Left Join neems_environment_index ne on ne.neems_ID = n.ID
Left JOIN (Select hpara.dul_Concept_s, dc.dul_Entity_o, hpara.dul_Parameter_o, hpara.neem_id
        From dul_hasParameter as hpara
            INNER JOIN dul_classifies as dc
                ON dc.dul_Concept_s = hpara.dul_Parameter_o and dc.neem_id = hpara.neem_id
        ) as hpara
On hpara.dul_Concept_s = task.dul_Task_o and hpara.neem_id = task.neem_id
Left JOIN rdf_type as  parameter_type
ON hpara.dul_Entity_o = parameter_type.s and parameter_type.o != 'owl:NamedIndividual' and hpara.neem_id = parameter_type.neem_id
Left JOIN dul_hasParticipant as task_hp
ON task_hp.dul_Event_s = task.dul_Action_s and task_hp.neem_id = task.neem_id
Left JOIN rdf_type as  participant_type
ON task_hp.dul_Object_o = participant_type.s and participant_type.o != 'owl:NamedIndividual'
and participant_type.neem_id = task_hp.neem_id
Left join dul_satisfies as t_sat
ON task.dul_Action_s = t_sat.dul_Situation_s
Order by task_ib.o;