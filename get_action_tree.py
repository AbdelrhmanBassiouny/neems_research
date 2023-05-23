from sqlalchemy import create_engine, Engine, Connection, text
from typing import Optional, Callable, Tuple, List, Dict
import argparse
import os
import pandas as pd
from jpt.variables import infer_from_dataframe
import jpt.trees
import numpy as np
import jpt.variables
from time import time
import logging
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


def get_value_from_sql(table_name: str, engine:Engine, col_names: Optional[list]=None, col_value_pairs: Optional[dict]=None) -> list:
        """Retrieve a list of values from a table column .

        Args:
            table_name (str): [The name of the table to retrieve the values from.]
            engine (Engine): [The sqlalchemy engine to use to connect to the database.]
            col_name (str, optional): [The name of the column to retrieve the values from.]. Defaults to 'ID'.
            col_value_pairs (dict, optional): [each pair is a column name and a value to search for in that column.]. Defaults to None.

        Returns:
            [list]: [list of values that match the search criteria.]
            
        """
        if col_names is None:
            col_names = ['*']
        col_names_str = ', '.join(col_names)
        if col_value_pairs is None:
            sql_cmd = f"SELECT {col_names_str} FROM {table_name};"
        else:
            sql_cmd = f"SELECT {col_names_str} FROM {table_name} WHERE "
            for i, (k, v) in enumerate(col_value_pairs.items()):
                sql_cmd += f"{k} = '{v}'"
                if i != len(col_value_pairs)-1:
                    sql_cmd += ' AND '
                else:
                    sql_cmd += ';'
        with engine.connect() as conn:
            try:
                result = conn.execute(text(sql_cmd))
                result = result.fetchall()
                if len(result[0]) == 1:
                    result = [x[0] for x in result]
            except:
                result = []
        return result


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--sql_username', '-su', help='SQL username')
    parser.add_argument('--sql_password', '-sp', help='SQL password')
    parser.add_argument('--sql_database', '-sd', help='SQL database name', required=False)
    parser.add_argument('--sql_host', '-sh', default="localhost", help='SQL host name')
    parser.add_argument('--sql_uri', '-suri', type=str, default=None, help='SQL URI this replaces the other SQL arguments')
    args = parser.parse_args()
    sql_username = args.sql_username
    sql_password = args.sql_password
    sql_database = args.sql_database
    sql_host = args.sql_host
    sql_uri = args.sql_uri
    sql_uri = os.environ.get('LOCAL_SQL_URL')
    # Create SQL engine
    if sql_uri is not None:
        SQL_URI = sql_uri
    else:
        SQL_URI = f"mysql+pymysql://{sql_username}:{sql_password}@{sql_host}/{sql_database}?charset=utf8mb4"
    engine = create_engine(SQL_URI)

    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    # Read sql file
    with open('tasks_subtasks_and_params.sql', 'r') as f:
        sql_cmd = f.read()

    with engine.connect() as conn:
        df = pd.read_sql(text(sql_cmd), conn)
    for cname in ['task', 'subtask']:
        df[f'{cname}_duration'] = np.abs(np.maximum(0, df[f'{cname}_end']) - np.maximum(0, df[f'{cname}_start']))
        df.drop(columns=[f'{cname}_start', f'{cname}_end'], inplace=True)
    print(df.head())

    # Get all subtasks for each task from the dataframe.
    df['prev_prev_prev_subtask_type'] = 'None'
    df['prev_prev_subtask_type'] = 'None'
    df['prev_subtask_type'] = 'None'
    df['next_subtask_type'] = 'None'
    df['prev_task_type'] = 'None'
    df['next_task_type'] = 'None'
    # df['prev_subtask_type'][0] = 'ABC'
    # print(df['prev_subtask_type'][0])
    # print(df[0]['prev_subtask_type'])
    # exit()

    # print(df['task_type'].unique())
    # print(len(df['task_type'].unique()))
    # print(df['subtask_type'].unique())
    # print(len(df['subtask_type'].unique()))
    # print(df['participant_type'].unique())
    # print(len(df['participant_type'].unique()))
    # exit()
    load = False
    if not load:
        task_subtask_dict = dict()
        start_time = time()
        for neem_id in df['neem_id'].unique():
            task_subtask_dict[neem_id] = dict()
            neem_indicies = df['neem_id'] == neem_id
            all_tasks = df[neem_indicies]['task'].unique()
            prev_task_indicies = None
            for j, task in enumerate(all_tasks):
                task_indicies = neem_indicies & (df['task'] == task)
                if prev_task_indicies is not None:
                    # print(df['task_type'][prev_task_indicies].values)
                    df['prev_task_type'][task_indicies] = df['task_type'][prev_task_indicies].values[0]
                    df['next_task_type'][prev_task_indicies] = df['task_type'][task_indicies].values[0]
                # prev_prev_subtask_indicies = prev_task_indicies
                prev_task_indicies = task_indicies
                task_subtask_dict[neem_id][task] = df['subtask'][task_indicies].unique().tolist()
                # for subtask in task_subtask_dict[neem_id][task]:
                #     time_start = df[(df['neem_id'] == neem_id) & (df['task'] == task) & (df['subtask'] == subtask)]['subtask_start'].values[0]
                #     time_end = df[(df['neem_id'] == neem_id) & (df['task'] == task) & (df['subtask'] == subtask)]['subtask_end'].values[0]
                #     print(time_start, time_end)
                for i, subtask in enumerate(task_subtask_dict[neem_id][task]):
                    df_task_subtask_indicies = task_indicies & (df['subtask'] == subtask)
                    if i > 0:
                        prev_subtask_indicies = task_indicies & (df['subtask'] == task_subtask_dict[neem_id][task][i-1])
                        df['prev_subtask_type'][df_task_subtask_indicies] = df['subtask_type'][prev_subtask_indicies].values[0]
                        # df_task_subtask['prev_subtask'] = task_subtask_dict[neem_id][task][i-1]
                    if i > 1:
                        # prev_prev_subtask_indicies = prev_subtask_indicies
                        df['prev_prev_subtask_type'][df_task_subtask_indicies] = df['prev_subtask_type'][prev_subtask_indicies].values[0]
                    if i > 2:
                        df['prev_prev_prev_subtask_type'][df_task_subtask_indicies] = df['prev_prev_subtask_type'][prev_subtask_indicies].values[0]
                    if i != len(task_subtask_dict[neem_id][task])-1:
                        next_subtask_indicies = task_indicies & (df['subtask'] == task_subtask_dict[neem_id][task][i+1])
                        df['next_subtask_type'][df_task_subtask_indicies] = df['subtask_type'][next_subtask_indicies].values[0]
                        # df_task_subtask['next_subtask'] = task_subtask_dict[neem_id][task][i+1]
                    # print(df_task_subtask)
                # print(task)
                # print(df['next_subtask_type'][task_indicies].values)
                # print("=====================================")
                # print(task_subtask_dict[neem_id][task])
            # print(df['next_task_type'][neem_indicies].values)
            # print("=====================================")
        print(f"Time to get task subtask dict: {time() - start_time}")
        for cname in ['task', 'subtask', 'neem_id', 'task_duration', 'subtask_duration', 'participant']:
            df.drop(columns=f'{cname}', inplace=True)
        df.fillna(value='None', inplace=True)
        # new_df = df[(df['next_subtask_type'] == 'None') & (df['next_task_type'] == 'None')]
        # print(new_df.head())
        # exit()
        # print(task_subtask_dict)
        # print(df.head())
        # exit()
        variables = infer_from_dataframe(df, scale_numeric_types=False)
        print(variables)
        model = jpt.trees.JPT(variables, min_samples_leaf=0.00005)
        model.fit(df)
        print("number_of_leaves = ", len(model.leaves))
        print(model.priors['environment'])
        # model.plot(directory="/tmp/neem_action_tree", plotvars=variables)
        model.save("/tmp/neem_action_tree.jpt")

        print("model size", model.number_of_parameters())
        evidence = dict()
        evidence["task_type"] = {'soma:PhysicalTask', 'soma:Transporting', 'soma:Accessing',
     'soma:Navigating', 'soma:MovingTo', 'soma:Opening', 'soma:LookingAt',
     'soma:LookingFor', 'soma:Perceiving', 'soma:Fetching', 'soma:PickingUp',
     'soma:Delivering', 'soma:Placing', 'soma:Sealing', 'soma:Closing'}
        evidence["subtask_type"] = {'soma:SettingGripper', 'soma:Releasing', 'soma:Gripping',
    'soma:AssumingArmPose', 'soma:PickingUp', 'soma:Placing',
    'soma:Navigation', 'soma:Navigating', 'soma:Transporting',
    'soma:LookingAt', 'soma:Detecting', 'soma:Opening', 'soma:Closing'} # Current PyCRAM action types
        evidence["prev_prev_prev_subtask_type"] = evidence["subtask_type"].union({'None'})
        evidence["prev_prev_subtask_type"] = evidence["subtask_type"].union({'None'})
        evidence["prev_subtask_type"] = evidence["subtask_type"].union({'None'})
        evidence["next_subtask_type"] = evidence["subtask_type"].union({'None'})
        evidence["prev_task_type"] = evidence["task_type"].union({'None'})
        evidence["next_task_type"] = evidence["task_type"].union({'None'})
        evidence["environment"] = {"Kitchen"}
        evidence["participant_type"] = {'None', 'soma:Bowl', 'soma:Milk',
                                        'soma:Plate', 'soma:Spoon', 'soma:Cereal',
                                        'soma:Fork', 'soma:Cup'}
    else:
        # model = jpt.trees.JPT.load("/tmp/neem_action_tree.jpt")
        # model = jpt.trees.JPT.load("neem_action_tree_looper_16_5_2023.jpt")
        model = jpt.trees.JPT.load("neem_action_tree_2_prev_looper_19_5_2023.jpt")
        evidence = dict()
        # evidence['task_type'] = {'soma:PhysicalTask'}
        evidence['subtask_type'] = {'soma:LookingAt'}
        evidence['participant_type'] = {'None', 'soma:Bowl', 'soma:Milk',
                                        'soma:Plate', 'soma:Spoon', 'soma:Cereal',
                                        'soma:Fork', 'soma:Cup'}
        # evidence['subtask_param'] = {'None'}
        # evidence['subtask_state'] = {'soma:ExecutionState_Succeeded'}
        # evidence['task_state'] = {'soma:ExecutionState_Succeeded'}
        # evidence['activity'] = {'Kitchen activity'}
        # evidence['environment'] = {'Kitchen'}
        evidence['prev_subtask_type'] = {'soma:PickingUp'}
        # evidence['prev_prev_subtask_type'] = {'soma:LookingAt'}
        # evidence['next_subtask_type'] = {'soma:Closing'}
        # evidence['prev_task_type'] = {'None'}
        # evidence['next_task_type'] = {'soma:MovingTo'}

    mpe, likelihood = model.mpe(model.bind(evidence))
    # print(mpe)
    top_task = Node(mpe[0]['prev_task_type'])
    task = Node(mpe[0]['task_type'], parent=top_task)
    prev_subtask = Node(mpe[0]['prev_subtask_type'], parent=task)
    # subtask = Node(str(mpe[0]['subtask_type']), parent=task)
    next_task = task
    # task_tree = {Node():[],
    #              Node(str(mpe[0]['task_type']), parent=Node(str(mpe[0]['prev_task_type']))):
    #              [Node(str(mpe[0]['prev_subtask_type'])), Node(str(mpe[0]['subtask_type']))]}
    i = 0
    j = 1
    k = 1
    while not (('None' in mpe[0]['next_task_type'] and 'None' in mpe[0]['next_subtask_type']) or i > 30):
        print(mpe[0])
        if len(mpe[0]['next_subtask_type']) > 1:
            print(mpe[0]['next_subtask_type'])
            exit()
        if 'None' in mpe[0]['next_subtask_type']:
            mpe[0]['prev_task_type'] = mpe[0]['task_type']
            mpe[0]['task_type'] = mpe[0]['next_task_type']
            next_task = Node(str(mpe[0]['task_type']) + str(j), parent=next_task.parent)
            del mpe[0]['next_task_type']
            j += 1
            mpe[0]['prev_prev_prev_subtask_type'] = 'None'
            mpe[0]['prev_prev_subtask_type'] = 'None'
            mpe[0]['prev_subtask_type'] = 'None'
            del mpe[0]['subtask_type']
        else:
            mpe[0]['prev_prev_prev_subtask_type'] = mpe[0]['prev_prev_subtask_type']
            mpe[0]['prev_prev_subtask_type'] = mpe[0]['prev_subtask_type']
            mpe[0]['prev_subtask_type'] = mpe[0]['subtask_type']
            mpe[0]['subtask_type'] = mpe[0]['next_subtask_type']
        del mpe[0]['next_subtask_type']
        mpe[0]['participant_type'] = {'None', 'soma:Bowl', 'soma:Milk',
                                        'soma:Plate', 'soma:Spoon', 'soma:Cereal',
                                        'soma:Fork', 'soma:Cup'}
        del mpe[0]['subtask_param']
        del mpe[0]['subtask_state']
        mpe[0]['activity'] = {'Kitchen activity'}
        mpe[0]['environment'] = {'Kitchen'}
        del mpe[0]['task_state']
        del mpe[0]['neem_name']
        del mpe[0]['neem_desc']
        mpe, likelihood = model.mpe(mpe[0])
        if 'None' not in mpe[0]['subtask_type']:
            subtask = Node(str(mpe[0]['subtask_type']) + str(k), parent=next_task)
            k += 1
        i += 1
    
    for pre, fill, node in RenderTree(top_task):
        print("%s%s" % (pre, node.name))
    DotExporter(top_task).to_picture("loopy_tree.png")
        

    # Get all the action tree IDs
    # action_tree = get_value_from_sql('dul_precedes', engine,
    #                                  col_names=['dul_Entity_s', 'dul_Entity_o'],
    #                                  col_value_pairs={'neem_id': '633819942a459501ef4d4209'})
    # action_tree = get_value_from_sql('dul_precedes', engine,
    #                                  col_names=['dul_Entity_s', 'dul_Entity_o'],
    #                                  col_value_pairs={'neem_id': '633819942a459501ef4d4209'})
    # print(action_tree)
    
    