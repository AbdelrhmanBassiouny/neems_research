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
import matplotlib.pyplot as plt
import pickle
import plotly.express as px
import plotly.graph_objects as go




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

def infer_and_fit_model_from_df(df, remove_subtasks=False):
    for cname in ['task', 'subtask', 'neem_id', 'task_duration', 'subtask_duration', 'participant', 'neem_name', 'neem_desc']:
        if cname in df.columns:
            df.drop(columns=f'{cname}', inplace=True)
    if remove_subtasks:
        for col in df.columns:
            if 'subtask' in col:
                df.drop(columns=col, inplace=True)
    variables = infer_from_dataframe(df, scale_numeric_types=False)
    # print(variables)
    model = jpt.trees.JPT(variables, min_samples_leaf=0.00005)
    model.fit(df)
    return model, variables

def get_sql_query_from_dict(dict, table_name):
    sql_cmd = f"SELECT * FROM {table_name} WHERE "
    for i, (k, v1) in enumerate(dict.items()):
        for j, v in enumerate(v1):
            if j == 0:
                sql_cmd += f"{k} in ('{v}'"
            else:
                sql_cmd += f", '{v}'"
            if j == len(v1)-1:
                sql_cmd += ')'
        if i != len(dict)-1:
            sql_cmd += ' AND '
        else:
            sql_cmd += ';'
    print(sql_cmd)
    return text(sql_cmd)

def filter_df_from_dict(df, dict, return_mode=False):
    df_filtered = df.copy()
    for k, v in dict.items():
        for i, e in enumerate(v):
            if i == 0:
                cond = df_filtered[k] == e
            else:
                cond = cond | (df_filtered[k] == e)
        # print(np.where(cond)[0])
        # print(k, v)
        df_filtered = df_filtered[cond]
    if return_mode:
        df_filtered.to_csv('df_filtered.csv')
        return df_filtered, get_row_mode(df_filtered)
    else:
        return df_filtered

def get_row_mode(df):
    df_np = df.to_numpy().astype(str)
    cols = df.columns
    unique_values, counts = np.unique(df_np, return_counts=True, axis=0)
    most_frequent_values = unique_values[np.argmax(counts)]
    return {cols[i]: {most_frequent_values[i]} for i in range(len(cols))}

def get_task_tree(current_data, model=None, tree_name='task_tree', use_dataframe=False, use_sql=False, engine=None):
    top_task = Node(current_data['prev_1_task_type'])
    task = Node(current_data['task_type'], parent=top_task)
    # prev_subtask = Node(current_data['prev_subtask_type'], parent=task)
    subtask = Node(str(current_data['subtask_type']), parent=task)
    next_task = task

    task_count = 0
    j = 1
    k = 1
    while not (('None' in current_data['next_task_type'] and 'None' in current_data['next_subtask_type']) or task_count > 40):
        print(current_data)
        if len(current_data['next_subtask_type']) > 1:
            print(current_data['next_subtask_type'])
            exit()
        if 'None' in current_data['next_subtask_type']:
            for i in range(n_prev_tasks, 1, -1):
                current_data[f'prev_{i}_task_type'] = current_data[f'prev_{i-1}_task_type']
            current_data['prev_1_task_type'] = current_data['task_type']
            current_data['task_type'] = current_data['next_task_type']
            next_task = Node(str(current_data['task_type']) + str(j), parent=next_task.parent)
            j += 1
            del current_data['next_task_type']
            for i in range(1, n_prev_subtasks + 1):
                current_data[f'prev_{i}_subtask_type'] = {'None'}
            del current_data['subtask_type']
        else:
            for i in range(n_prev_subtasks, 1, -1):
                current_data[f'prev_{i}_subtask_type'] = current_data[f'prev_{i-1}_subtask_type']
            current_data['prev_1_subtask_type'] = current_data['subtask_type']
            current_data['subtask_type'] = current_data['next_subtask_type']

        del current_data['next_subtask_type']
        current_data['participant_type'] = {'None', 'soma:Bowl', 'soma:Milk',
                                        'soma:Plate', 'soma:Spoon', 'soma:Cereal',
                                        'soma:Fork', 'soma:Cup'}
        del current_data['subtask_param']
        del current_data['subtask_state']
        del current_data['task_state']
        # del current_data['neem_name']
        # del current_data['neem_desc']
        # current_data['activity'] = {'Kitchen activity'}
        # current_data['environment'] = {'Kitchen'}
        
        if use_dataframe:
            df_filtered, df_mode = filter_df_from_dict(df, current_data, return_mode=True)
            print(df_mode)
            df_mode = df_mode.to_dict(orient='list')
            df_mode = {k: {v[0]} for k, v in df_mode.items()}
            current_data = df_mode
        elif use_sql:
            sql_cmd = get_sql_query_from_dict(current_data, 'task_tree')
            with engine.connect() as conn:
                dataframe = pd.read_sql_query(sql_cmd, conn)
            # print(dataframe)
            current_data = get_row_mode(dataframe)
        else:
            mpe, likelihood = model.mpe(current_data)
            current_data = mpe[0]
        if 'None' not in current_data['subtask_type']:
            subtask = Node(str(current_data['subtask_type']) + str(k), parent=next_task)
            k += 1
        task_count += 1

    for pre, fill, node in RenderTree(top_task):
        print("%s%s" % (pre, node.name))
    DotExporter(top_task).to_picture(f"{tree_name}.png")


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

    load_df = True
    load_from_sql = not load_df
    save_df = True
    infer_from_df = True
    n_prev_subtasks = 3
    n_prev_tasks = 3
    n_top_tasks = 2
    n_parent_tasks = 1
    plot_tasks = False
    
    if plot_tasks:
        df = pd.read_pickle('df.pkl')
        print(df.head())
        use_type = True
        subtasks = df['subtask_type'].unique()
        colors_dict = {subtasks[i]: f'C{i}' for i in range(len(subtasks))}
        for neem_id in df['neem_id'].unique():
            curr_df = df[df['neem_id'] == neem_id]

            start = dict()
            end = dict()
            lineoffsets = dict()
            linelengths = dict()
            st_data = curr_df['subtask_type'].values if use_type else curr_df['subtask'].values
            t_data = curr_df['task_type'].values if use_type else curr_df['task'].values
            ut_data = pd.DataFrame(t_data)[0].unique()
            # print(curr_df.head())
            fig = go.Figure()
            neg_vals = curr_df[f'subtask_start'] < 0
            curr_df[f'subtask_start'][neg_vals] = curr_df[f'task_start'][neg_vals]
            neg_vals = curr_df[f'subtask_end'] < 0
            curr_df[f'subtask_end'][neg_vals] = curr_df[f'task_end'][neg_vals]

            fig = px.timeline(curr_df, x_start=pd.to_datetime(curr_df[f'task_start'], unit='s'),
                                x_end=pd.to_datetime(curr_df[f'task_end'], unit='s'),
                                    y=f'task_type',
                                    color=f'task_type',
                                    hover_data={'subtask':True, 'task':True, 'participant_type':True},
                                    # text=f'subtask_type',
                                    title=f"tasks for {curr_df['neem_name'].values[0]}")
            close_vals = [1]
            while len(close_vals) > 0:
                diff = np.diff(curr_df[f'task_start'])
                close_vals = np.where(diff < 5)[0]
                for i in range(len(close_vals)):
                    # delete the close task
                    curr_df.drop(curr_df.index[close_vals[i]+1], inplace=True)
                    break
            fig.update_xaxes(tickvals=pd.to_datetime(curr_df[f'task_start'], unit='s'), tickformat='%M:%S')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
            # for s in curr_df['task_start'].unique():
            #     fig.add_vline(x=pd.to_datetime(s, unit='s'))
            # fig.update_yaxes(autorange="reversed")          #if not specified as 'reversed', the tasks will be listed from bottom up       
            # fig.data[1].width=0.5 # update the width of the 'Actual' schedule bars (the second trace of the figure)
            # for i in range(1,len(fig.data)):
            #     fig.data[i].width = 0.5
            fig.show()
            # exit()
        exit()
    else:
        # Read sql file
        with open('tasks_subtasks_and_params.sql', 'r') as f:
            sql_cmd = f.read()

        with engine.connect() as conn:
            df = pd.read_sql(text(sql_cmd), conn)
        for cname in ['task', 'subtask']:
            df[f'{cname}_duration'] = np.abs(np.maximum(0, df[f'{cname}_end']) - np.maximum(0, df[f'{cname}_start']))
            # df.drop(columns=[f'{cname}_start', f'{cname}_end'], inplace=True)
        print(df.head())

        # Get all subtasks for each task from the dataframe.
        # for i in range(1, n_prev_subtasks + 1):
        #     df[f'prev_{i}_subtask_type'] = df.groupby('task_type')['subtask_type'].shift(i)
        for i in range(1, n_prev_subtasks + 1):
            df[f'prev_{i}_subtask_type'] = 'None'
        df['next_subtask_type'] = 'None'
        for i in range(1, n_prev_tasks + 1):
            df[f'prev_{i}_task_type'] = 'None'
        df['next_task_type'] = 'None'
        for i in range(1, n_top_tasks + 1):
            df[f'top_{i}_task_type'] = 'None'
        for i in range(1, n_parent_tasks + 1):
            df[f'parent_{i}_task_type'] = 'None'

    evidence = dict()
    evidence["task_type"] = {'soma:PhysicalTask', 'soma:Transporting', 'soma:Accessing',
    'soma:Navigating', 'soma:MovingTo', 'soma:Opening', 'soma:LookingAt',
    'soma:LookingFor', 'soma:Perceiving', 'soma:Fetching', 'soma:PickingUp',
    'soma:Delivering', 'soma:Placing', 'soma:Sealing', 'soma:Closing'}
    evidence["subtask_type"] = {'soma:SettingGripper', 'soma:Releasing', 'soma:Gripping',
'soma:AssumingArmPose', 'soma:PickingUp', 'soma:Placing',
'soma:Navigation', 'soma:Navigating', 'soma:Transporting',
'soma:LookingAt', 'soma:Detecting', 'soma:Opening', 'soma:Closing'} # Current PyCRAM action types
    for i in range(1, n_prev_subtasks + 1):
        evidence[f'prev_{i}_subtask_type'] = evidence["subtask_type"].union({'None'})
    evidence["next_subtask_type"] = evidence["subtask_type"].union({'None'})
    for i in range(1, n_prev_tasks + 1):
        evidence[f'prev_{i}_task_type'] = evidence["task_type"].union({'None'})
    evidence["next_task_type"] = evidence["task_type"].union({'None'})
    evidence["environment"] = {"Kitchen"}
    evidence["participant_type"] = {'None', 'soma:Bowl', 'soma:Milk',
                                    'soma:Plate', 'soma:Spoon', 'soma:Cereal',
                                    'soma:Fork', 'soma:Cup'}
    evidence = dict()
    # evidence['task_type'] = {'soma:PhysicalTask'}
    evidence['subtask_type'] = {'soma:LookingAt'}
    evidence['participant_type'] = {'soma:Bowl', 'soma:Milk',
                                    'soma:Plate', 'soma:Spoon', 'soma:Cereal',
                                    'soma:Fork', 'soma:Cup'}
    # evidence['subtask_param'] = {'None'}
    # evidence['subtask_state'] = {'soma:ExecutionState_Succeeded'}
    # evidence['task_state'] = {'soma:ExecutionState_Succeeded'}
    # evidence['activity'] = {'Kitchen activity'}
    # evidence['environment'] = {'Kitchen'}
    # evidence['prev_subtask_type'] = {'soma:PickingUp'}
    # evidence['prev_prev_subtask_type'] = {'soma:LookingAt'}
    # evidence['next_subtask_type'] = {'soma:Closing'}
    # evidence['prev_task_type'] = {'None'}
    # evidence['next_task_type'] = {'soma:MovingTo'}

    with open('evidence.pkl', 'wb') as f:
        pickle.dump(evidence, f)

    if load_df:
        model, variables = infer_and_fit_model_from_df(df)
        print(df)
    elif load_from_sql:
        task_subtask_dict = dict()
        start_time = time()
        for neem_id in df['neem_id'].unique():
            task_subtask_dict[neem_id] = dict()
            neem_indicies = df['neem_id'] == neem_id
            all_tasks = df[neem_indicies]['task'].unique()
            prev_task_indicies = None
            prev_task_end_time = 0
            for j, task in enumerate(all_tasks):

                task_indicies = neem_indicies & (df['task'] == task)
                task_subtask_dict[neem_id][task] = df['subtask'][task_indicies].unique().tolist()

                if j < n_top_tasks:
                    task_type = df['task_type'][task_indicies].values[0]
                    df[f'top_{j+1}_task_type'][neem_indicies] = task_type
                
                # Subtasks
                for i, subtask in enumerate(task_subtask_dict[neem_id][task]):                        
                    df_task_subtask_indicies = task_indicies & (df['subtask'] == subtask)
                    if i > 0:
                        df['next_subtask_type'][prev_subtask_indicies] = df['subtask_type'][df_task_subtask_indicies].values[0]

                    p = 0
                    used_p = 0
                    done = False
                    while not done and i > 0:
                        subtask1_name = f'prev_{used_p+1}_subtask_type'
                        subtask2_name = 'subtask_type' if p == 0 else f'prev_{p}_subtask_type'
                        if df['subtask_end'][prev_subtask_indicies].values[0] <= df['subtask_start'][df_task_subtask_indicies].values[0] and \
                            used_p < n_prev_subtasks:
                            df[subtask1_name][df_task_subtask_indicies] = df[subtask2_name][prev_subtask_indicies].values[0]
                            used_p += 1
                        p += 1
                        if (used_p == n_prev_tasks) or p == i:
                            done = True
                    prev_subtask_indicies = df_task_subtask_indicies
                
                if prev_task_indicies is None:
                    prev_task_indicies = task_indicies
                    continue

                # Tasks
                done = False
                p = 0
                used_p = 0
                parent_i = 0
                while not done:
                    task1_name = f'prev_{used_p+1}_task_type'
                    task2_name = 'task_type' if p == 0 else f'prev_{p}_task_type'
                    if df['task_end'][prev_task_indicies].values[0] <= df['task_start'][task_indicies].values[0] and \
                        used_p < n_prev_tasks:
                        df[task1_name][task_indicies] = df[task2_name][prev_task_indicies].values[0]
                        used_p += 1
                    task1_name = f'parent_{parent_i+1}_task_type'
                    task2_name = 'task_type' if p == 0 else f'parent_{p}_task_type'
                    if df['task_end'][prev_task_indicies].values[0] >= df['task_end'][task_indicies].values[0] and \
                        df['task_start'][prev_task_indicies].values[0] <= df['task_start'][task_indicies].values[0] and \
                            parent_i < n_parent_tasks:
                        df[task1_name][task_indicies] = df[task2_name][prev_task_indicies].values[0]
                        # print(f"{task1_name} {df[task1_name][task_indicies].values[0]}")
                        # print(f"child {df['task_type'][task_indicies].values[0]}")
                        parent_i += 1
                    p += 1
                    if (used_p == n_prev_tasks and parent_i == n_parent_tasks) or p == j:
                        done = True

                df['next_task_type'][prev_task_indicies] = df['task_type'][task_indicies].values[0]
                prev_task_indicies = task_indicies

        # prev task type
        # df.groupby('neem_id').group
        # next task type
        # df['next_task_type'] = df.groupby('neem_id')['task_type'].shift(-1)
        print(f"Time to get task subtask dict: {time() - start_time}")
        # for i in range(1, n_prev_subtasks + 1):
        #     df[f'prev_{i}_subtask_type'] = df.groupby('task_type')['subtask_type'].shift(i)
        # df['next_subtask_type'] = df.groupby('task_type')['subtask_type'].shift(-1)
        # for cname in ['task', 'subtask', 'neem_id', 'task_duration', 'subtask_duration', 'participant', 'neem_name', 'neem_desc']:
            # df.drop(columns=f'{cname}', inplace=True)
        df.fillna(value='None', inplace=True)
        if save_df:
            df.to_pickle('df.pkl')

        model, variables = infer_and_fit_model_from_df(df)
        print("number_of_leaves = ", len(model.leaves))
        print(model.priors['environment'])
        # model.plot(directory="/tmp/neem_action_tree", plotvars=variables)
        model.save("/tmp/neem_action_tree.jpt")

        print("model size", model.number_of_parameters())

    else:
        # model = jpt.trees.JPT.load_from_sql("/tmp/neem_action_tree.jpt")
        # model = jpt.trees.JPT.load_from_sql("neem_action_tree_looper_16_5_2023.jpt")
        # model = jpt.trees.JPT.load_from_sql("neem_action_tree_2_prev_looper_19_5_2023.jpt")
        model = jpt.trees.JPT.load_from_sql("neem_action_tree_3_prev_23_5_2023.jpt")

    mpe, likelihood = model.mpe(model.bind(evidence))
    # print("mpe = ", mpe[0])

    if infer_from_df:
        df_filtered, df_mode = filter_df_from_dict(df, evidence, return_mode=True)
        df_filtered.apply(pd.value_counts).plot(kind='bar', subplots=True)
        df.to_sql('task_tree', engine, if_exists='replace', index=False)
        get_task_tree(df_mode, use_sql=True, tree_name='dataframe_tree', engine=engine)
        # get_task_tree(df_mode, use_dataframe=True, tree_name='dataframe_tree')
        plt.show()

    get_task_tree(mpe[0], model=model)

    
    