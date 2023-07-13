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
    for cname in ['task', 'subtask', 'neem_id', 'task_duration', 'subtask_duration', 'task_start', 'task_end',
                  'subtask_start', 'subtask_end', 'participant', 'neem_name', 'neem_desc', 'created_by', 'activity']:
        if cname in df.columns:
            df.drop(columns=f'{cname}', inplace=True, errors='ignore')
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

def set_color_shape(node):
    attrs = []
    attrs += [f'color={node.color}'] if hasattr(node, 'color') else []
    attrs += [f'shape={node.shape}'] if hasattr(node, 'shape') else []
    return ', '.join(attrs)

def get_task_tree(current_data,
                  model=None,
                  tree_name='task_tree',
                  use_dataframe=False,
                  use_sql=False,
                  engine=None,
                  df=None,
                  plot=False,
                  enforce_success=False):
    if n_top_tasks > 0:
        top_task = Node(current_data['top_1_task_type'])
    else:
        top_task = Node('None')
    task = None
    j = -n_prev_tasks
    for i in range(n_prev_tasks, 0, -1):
        if 'None' in current_data[f'prev_{i}_task_type']:
            j += 1
            continue
        if task is None:
            task = Node(str(current_data[f'prev_{i}_task_type']) + f" {j}", parent=top_task)
        else:
            task = Node(str(current_data[f'prev_{i}_task_type']) + f" {j}", parent=task.parent)
        if f'prev_{i}_task_state' in current_data:
            state = Node(str(current_data[f'prev_{i}_task_state']) + f" {j}", parent=task)
            if 'soma:ExecutionState_Failed' in current_data[f'prev_{i}_task_state']:
                task.color = 'red'
                state.color = 'red'
            if f'prev_{i}_task_failure_type' in current_data:
                if 'None' not in current_data[f'prev_{i}_task_failure_type']:
                    failure_type = Node(str(current_data[f'prev_{i}_task_failure_type']) + f" {j}", parent=state)
                    failure_type.color = 'red'
        j += 1

    if task is not None:
        task = Node(str(current_data['task_type']) + f" {j}", parent=task.parent)
    else:
        task = Node((current_data['task_type']) + f" {j}", parent=top_task)
    if f'task_state' in current_data:
        state = Node(str(current_data[f'task_state']) + f" {j}", parent=task)
        if f'task_failure_type' in current_data:
            if 'None' not in current_data[f'task_failure_type']:
                failure_type = Node(str(current_data[f'task_failure_type']) + f" {j}", parent=state)
                task.color = 'red'
                state.color = 'red'
                failure_type.color = 'red'
    j += 1
    if use_subtasks:
        # prev_subtask = Node(current_data['prev_subtask_type'], parent=task)
        subtask = Node(str(current_data['subtask_type']), parent=task)
    next_task = task
    task_count = 0

    if use_subtasks:
        cond = lambda x: not (('None' in x['next_task_type'] and 'None' in x['next_subtask_type']) or task_count > 40)
    else:
        if n_parent_tasks > 0:
            cond = lambda x: not (('None' in x['next_task_type'] and 'None' in x['parent_1_task_type']) or task_count > 40)
        else:
            cond = lambda x: not (('None' in x['next_task_type']) or task_count > 60)

    k = 1
    dataframe = df
    while cond(current_data):
        print(current_data)
        if use_subtasks:
            if len(current_data['next_subtask_type']) > 1:
                print(current_data['next_subtask_type'])
                exit()
        if use_subtasks:
            change_task_cond = 'None' in current_data['next_subtask_type']
        else:
            change_task_cond = True 
        if change_task_cond:
            # Change the previous tasks and their attributes
            for i in range(n_prev_tasks, 1, -1):
                current_data[f'prev_{i}_task_type'] = current_data[f'prev_{i-1}_task_type']
                if attributes is not None:
                    for attrib in attributes:
                        current_data[f'prev_{i}_task_{attrib}'] = current_data[f'prev_{i-1}_task_{attrib}']
            current_data['prev_1_task_type'] = current_data['task_type']
            if attributes is not None:
                for attrib in attributes:
                    current_data[f'prev_1_task_{attrib}'] = current_data[f'task_{attrib}']
            # If next task is not None, change current task and its attributes to the next task and its attributes
            if 'None' not in current_data['next_task_type']:
                current_data['task_type'] = current_data['next_task_type']
                if attributes is not None:
                    for attrib in attributes:
                        current_data[f'task_{attrib}'] = current_data[f'next_task_{attrib}']
                next_task = Node(str(current_data['task_type']) + f" {j}", parent=next_task.parent)
                if f'task_state' in current_data and 'state' in attributes:
                    state = Node(str(current_data[f'task_state']) + f" {j}", parent=next_task)
                    if f'task_failure_type' in current_data and 'failure_type' in attributes:
                        if 'None' not in current_data[f'task_failure_type']:
                            failure_type = Node(str(current_data[f'task_failure_type']) + f" {j}", parent=state)
                            next_task.color = 'red'
                            state.color = 'red'
                            failure_type.color = 'red'
                j += 1
            # If next task is None, change current task and its attributes to the parent task and its attributes
            else:
                current_data['task_type'] = current_data['parent_1_task_type']
                if attributes is not None:
                    for attrib in attributes:
                        current_data[f'task_{attrib}'] = current_data[f'parent_1_task_{attrib}']
                for i in range(1, n_prev_tasks + 1):
                    del current_data[f'prev_{i}_task_type']
                    if attributes is not None:
                        for attrib in attributes:
                            del current_data[f'prev_{i}_task_{attrib}']
            del current_data['next_task_type']
            if attributes is not None:
                for attrib in attributes:
                    del current_data[f'next_task_{attrib}']
            for i in range(1, n_parent_tasks + 1):
                del current_data[f'parent_{i}_task_type']
                if attributes is not None:
                    for attrib in attributes:
                        del current_data[f'parent_{i}_task_{attrib}']
            if use_subtasks:
                for i in range(1, n_prev_subtasks + 1):
                    current_data[f'prev_{i}_subtask_type'] = {'None'}
                    if attributes is not None:
                        for attrib in attributes:
                            current_data[f'prev_{i}_subtask_{attrib}'] = {'None'}
                del current_data['subtask_type']
                if attributes is not None:
                    for attrib in attributes:
                        del current_data[f'subtask_{attrib}']
        else:
            for i in range(n_prev_subtasks, 1, -1):
                current_data[f'prev_{i}_subtask_type'] = current_data[f'prev_{i-1}_subtask_type']
                if attributes is not None:
                    for attrib in attributes:
                        current_data[f'prev_{i}_subtask_{attrib}'] = current_data[f'prev_{i-1}_subtask_{attrib}']
            current_data['prev_1_subtask_type'] = current_data['subtask_type']
            if attributes is not None:
                for attrib in attributes:
                    current_data[f'prev_1_subtask_{attrib}'] = current_data[f'subtask_{attrib}']
            current_data['subtask_type'] = current_data['next_subtask_type']
            if attributes is not None:
                for attrib in attributes:
                    current_data[f'subtask_{attrib}'] = current_data[f'next_subtask_{attrib}']

        if use_subtasks:
            del current_data['next_subtask_type']
            current_data['participant_type'] = {'None', 'soma:Bowl', 'soma:Milk',
                                            'soma:Plate', 'soma:Spoon', 'soma:Cereal',
                                            'soma:Fork', 'soma:Cup'}
            if attributes is not None:
                for attrib in attributes:
                    del current_data[f'next_subtask_{attrib}']

        if attributes is not None:
            for attrib in all_attributes:
                if attrib not in attributes:
                    if f'task_{attrib}' in current_data:
                        del current_data[f'task_{attrib}']
                    if f'subtask_{attrib}' in current_data:
                        del current_data[f'subtask_{attrib}']
        else:
            for attrib in all_attributes:
                if f'task_{attrib}' in current_data:
                    del current_data[f'task_{attrib}']
                if f'subtask_{attrib}' in current_data:
                    del current_data[f'subtask_{attrib}']
        
        # enforce success state
        if enforce_success:
            current_data['next_task_state'] = {'soma:ExecutionState_Succeeded', 'None'}
                
        # current_data['participant_type'].add('None')
        # del current_data['neem_name']
        # del current_data['neem_desc']
        # current_data['activity'] = {'Kitchen activity'}
        # current_data['environment'] = {'Kitchen'}
        prev_data = current_data.copy()
        print(current_data)
        if use_dataframe:
            dataframe, current_data = filter_df_from_dict(df, current_data, return_mode=True)
        elif use_sql:
            sql_cmd = get_sql_query_from_dict(current_data, 'task_tree')
            with engine.connect() as conn:
                dataframe = pd.read_sql_query(sql_cmd, conn)
            # print(dataframe)
            current_data = get_row_mode(dataframe)
        else:
            evidence = model.bind(current_data) if enforce_success else current_data
            mpe, likelihood = model.mpe(evidence)
            current_data = mpe[0].to_json() if enforce_success else mpe[0]
        
        if plot and (use_dataframe or use_sql) and cond(current_data):
            title = f"{j}_EVIDENCE::"
            for ti, (key, val) in enumerate(prev_data.items()):
                title += f"{key}={val}, "
                if ti % 2 == 0 and ti != 0:
                    title += '\n'

            plot_df_stats(dataframe, title=title, filename=j, save=True)

        if use_subtasks:
            if 'None' not in current_data['subtask_type']:
                subtask = Node(str(current_data['subtask_type']) + f" {k}", parent=next_task)
                k += 1
        task_count += 1

    for pre, fill, node in RenderTree(top_task):
        print("%s%s" % (pre, node.name))
    DotExporter(top_task, nodeattrfunc=set_color_shape).to_picture(f"{tree_name}.png")
    

def get_prev_task(df, current_task, current_start, current_end, heirarchy, task_type, task_attributes=None):
        task_idx = np.where(df[f'{task_type}'] == current_task)[0]
        if len(task_idx) == 0:
            raise ValueError(f"task {current_task} not found in dataframe")
        task_idx = task_idx[0]
        if heirarchy == 'prev':
            cond = df[f'{task_type}_end'][:task_idx] <= current_start
        elif heirarchy == 'parent':
            cond = (df[f'{task_type}_end'][:task_idx] >= current_end) & (df[f'{task_type}_start'][:task_idx] <= current_start)
        all_prev_indicies = np.where(cond)[0]
        if len(all_prev_indicies) == 0:
            return None, None, None
        prev_task_indicies = all_prev_indicies[-1]
        prev_task = df[f'{task_type}'][prev_task_indicies]
        prev_task_type = df[f'{task_type}_type'][prev_task_indicies]
        prev_task_attributes = None
        if task_attributes is not None:
            prev_task_attributes = {attrib:df[f'{task_type}_{attrib}'][prev_task_indicies] for attrib in task_attributes}
        return prev_task, prev_task_type, prev_task_attributes

def set_old_tasks(df, current_indicies, heirarchy, n_tasks, neem_indicies,
                  task_type='task',
                  df_to_modify=None,
                  set_attribures=None):
    """AI is creating summary for set_old_tasks

    Args:
        df ([type]): [end sorted task dataframe]
        task_indicies ([type]): [description]
        heirarchy ([type]): [description]
        df_to_modify ([type], optional): [description]. Defaults to None.
        modify_indicies ([type], optional): [description]. Defaults to None.
    """
    if df_to_modify is None:
        df_to_modify = df.copy(deep=False)
    current_start = df_to_modify[f'{task_type}_start'][current_indicies].values[0]
    current_end = df_to_modify[f'{task_type}_end'][current_indicies].values[0]
    current_task = df_to_modify[f'{task_type}'][current_indicies].values[0]
    current_task_type = df_to_modify[f'{task_type}_type'][current_indicies].values[0]
    current_task_attributes = {attrib: df_to_modify[f'{task_type}_{attrib}'][current_indicies].values[0]
                               for attrib in set_attribures}
    # find first old task
    prev_task, prev_task_type, prev_task_attributes = get_prev_task(df, current_task, current_start, current_end,
                                                                  heirarchy, task_type, task_attributes=set_attribures)
    if prev_task is None:
        return
    df_to_modify[f'{heirarchy}_1_{task_type}_type'][current_indicies] = prev_task_type
    if set_attribures is not None:
        for attrib in set_attribures:
            df_to_modify[f'{heirarchy}_1_{task_type}_{attrib}'][current_indicies] = prev_task_attributes[attrib]
    prev_task_indicies = (df_to_modify[f'{task_type}'] == prev_task)  & neem_indicies
    # set next task type
    if heirarchy == 'prev':
        if df_to_modify[f'next_{task_type}_type'][prev_task_indicies].values[0] == 'None':
            df_to_modify[f'next_{task_type}_type'][prev_task_indicies] = current_task_type
            if set_attribures is not None:
                for attrib in set_attribures:
                    df_to_modify[f'next_{task_type}_{attrib}'][prev_task_indicies] = current_task_attributes[attrib]
    # find all older tasks
    for i in range(2, n_tasks + 1):
        df_to_modify[f'{heirarchy}_{i}_{task_type}_type'][current_indicies] =\
              df_to_modify[f'{heirarchy}_{i-1}_{task_type}_type'][prev_task_indicies].values[0]
        if set_attribures is not None:
            for attrib in set_attribures:
                df_to_modify[f'{heirarchy}_{i}_{task_type}_{attrib}'][current_indicies] =\
                    df_to_modify[f'{heirarchy}_{i-1}_{task_type}_{attrib}'][prev_task_indicies].values[0]

def plot_df_stats(df, title='df_stats', filename='stats', save=False):
    ax = df.apply(pd.value_counts).plot(kind='bar', subplots=True, figsize=(30, 20), fontsize=15, rot=45)
    for a in ax:
        a.set_title(a.get_title(), fontsize=25)
        a.legend(fontsize=20)
    fig = ax[0].get_figure()
    fig.suptitle(title, fontsize=30)
    if save:
        fig.savefig(f"{filename}.png")


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

    use_subtasks = False
    use_participant = False
    attributes = None
    all_attributes = ['participant_type', 'participant', 'param', 'state', 'failure_type']
    attributes = ['participant_type', 'state']
    load_df = True
    load_from_sql = not load_df
    save_df = True
    infer_from_df = True
    enforce_success = False
    n_prev_subtasks = 0
    n_prev_tasks = 3
    n_top_tasks = 1
    n_parent_tasks = 1
    n_parent_subtasks = 0
    plot_tasks = False
    
    if plot_tasks:
        df = pd.read_pickle('df.pkl')
        print(df.head())
        use_type = True
        if use_subtasks:
            subtasks = df['subtask_type'].unique()
            colors_dict = {subtasks[i]: f'C{i}' for i in range(len(subtasks))}

        for neem_id in df['neem_id'].unique():
            curr_df = df[df['neem_id'] == neem_id]

            start = dict()
            end = dict()
            lineoffsets = dict()
            linelengths = dict()

            if use_subtasks:
                st_data = curr_df['subtask_type'].values if use_type else curr_df['subtask'].values
                neg_vals = curr_df[f'subtask_start'] < 0
                curr_df[f'subtask_start'][neg_vals] = curr_df[f'task_start'][neg_vals]
                neg_vals = curr_df[f'subtask_end'] < 0
                curr_df[f'subtask_end'][neg_vals] = curr_df[f'task_end'][neg_vals]

            t_data = curr_df['task_type'].values if use_type else curr_df['task'].values
            ut_data = pd.DataFrame(t_data)[0].unique()
            # print(curr_df.head())
            fig = go.Figure()

            fig = px.timeline(curr_df, x_start=pd.to_datetime(curr_df[f'task_start'], unit='s'),
                                x_end=pd.to_datetime(curr_df[f'task_end'], unit='s'),
                                    y=f'task_type',
                                    color=f'task_type',
                                    hover_data={'task':True, 'prev_1_task_type':True, 'parent_1_task_type':True, 'next_task_type':True},
                                    # hover_data={'subtask':True, 'task':True, 'participant_type':True},
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
    if load_df:
        df = pd.read_pickle('df.pkl')
        print(df.head())
    else:
        # Read sql file
        if use_subtasks:
            sql_query_file = 'tasks_subtasks_and_params.sql'
        elif use_participant:
            sql_query_file = 'tasks_with_participant.sql'
        elif attributes is not None:
            sql_query_file = 'tasks_all_attributes.sql'
        else:
            sql_query_file = 'tasks.sql'
        with open(sql_query_file, 'r') as f:
            sql_cmd = f.read()

        with engine.connect() as conn:
            df = pd.read_sql(text(sql_cmd), conn)
        task_types = ['task', 'subtask'] if use_subtasks else ['task']
        for cname in task_types:
            df[f'{cname}_duration'] = np.abs(np.maximum(0, df[f'{cname}_end']) - np.maximum(0, df[f'{cname}_start']))
            # df.drop(columns=[f'{cname}_start', f'{cname}_end'], inplace=True)
        print(df.head())

        # Get all subtasks for each task from the dataframe.
        if use_subtasks:
            for i in range(1, n_prev_subtasks + 1):
                df[f'prev_{i}_subtask_type'] = 'None'
            df['next_subtask_type'] = 'None'
            for i in range(1, n_parent_subtasks + 1):
                df[f'parent_{i}_subtask_type'] = 'None'
        for i in range(1, n_prev_tasks + 1):
            if attributes is not None:
                for attrib in attributes:
                    df[f'prev_{i}_task_{attrib}'] = 'None'
            df[f'prev_{i}_task_type'] = 'None'
        if attributes is not None:
            for attrib in attributes:
                df[f'next_task_{attrib}'] = 'None'
        df['next_task_type'] = 'None'
        for i in range(1, n_top_tasks + 1):
            df[f'top_{i}_task_type'] = 'None'
        for i in range(1, n_parent_tasks + 1):
            if attributes is not None:
                for attrib in attributes:
                    df[f'parent_{i}_task_{attrib}'] = 'None'
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
    evidence['top_1_task_type'] = {'soma:Transporting'}
    # evidence['prev_3_task_type'] = {'None'}
    evidence['task_type'] = {'soma:Opening'}
    # evidence['next_task_participant_type'] = {'soma:Milk'}
    # evidence['task_state'] = {'soma:ExecutionState_Failed'}
    # evidence['next_task_state'] = {'soma:ExecutionState_Succeeded', 'None'}
    # evidence['participant_type'] = {'soma:Bowl', 'soma:Milk',
    #                                 'soma:Plate', 'soma:Spoon', 'soma:Cereal',
    #                                 'soma:Fork', 'soma:Cup'}
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
        start_time = time()
        for neem_id in df['neem_id'].unique():
            neem_indicies = df['neem_id'] == neem_id
            all_tasks = df[neem_indicies]['task'].unique()
            # end sorted dataframe
            end_sorted_df = df[neem_indicies].sort_values(by=['task_end'], ascending=True, ignore_index=True)
            for j, task in enumerate(all_tasks):

                task_indicies = neem_indicies & (df['task'] == task)

                if j < n_top_tasks:
                    task_type = df['task_type'][task_indicies].values[0]
                    df[f'top_{j+1}_task_type'][neem_indicies] = task_type
                
                # Subtasks
                if use_subtasks:
                    for i, subtask in enumerate(df['subtask'][task_indicies].unique().tolist()):
                        if i > 0:
                            df_task_subtask_indicies = task_indicies & (df['subtask'] == subtask)
                            set_old_tasks(df[task_indicies], df_task_subtask_indicies, 'prev', n_prev_subtasks, neem_indicies, task_type='subtask', set_attribures=attributes)
                            if n_parent_subtasks > 0:
                                set_old_tasks(df[task_indicies], df_task_subtask_indicies, 'parent', n_parent_subtasks, neem_indicies, task_type='subtask', set_attribures=attributes)

                # Tasks
                if j > 0:
                    set_old_tasks(end_sorted_df, task_indicies, 'prev', n_prev_tasks, neem_indicies, task_type='task', df_to_modify=df, set_attribures=attributes)
                    if n_parent_tasks > 0:
                        set_old_tasks(df, task_indicies, 'parent', n_parent_tasks, neem_indicies, task_type='task', set_attribures=attributes)

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
        # model = jpt.trees.JPT.load("/tmp/neem_action_tree.jpt")
        # model = jpt.trees.JPT.load("neem_action_tree_looper_16_5_2023.jpt")
        # model = jpt.trees.JPT.load("neem_action_tree_2_prev_looper_19_5_2023.jpt")
        model = jpt.trees.JPT.load("neem_action_tree_3_prev_23_5_2023.jpt")
    
    print("Mean Log Likelihood = ", np.mean(np.log(model.likelihood(df))))
    # exit()
    mpe, likelihood = model.mpe(model.bind(evidence))
    # print("mpe = ", mpe[0])
    # print(model.bind(mpe[0].to_json()))
    # exit()

    if infer_from_df:
        plot_df_stats(df, title='df_stats', filename='df_stats', save=True)
        df.to_sql('task_tree', engine, if_exists='replace', index=False)
        df_filtered, df_mode = filter_df_from_dict(df, evidence, return_mode=True)
        plot_df_stats(df_filtered, title=f"1_{evidence}", filename='1', save=True)
        # df.to_sql('task_tree', engine, if_exists='replace', index=False)
        # get_task_tree(df_mode, use_sql=True, tree_name='dataframe_tree', engine=engine, enforce_success=enforce_success)
        get_task_tree(df_mode, use_dataframe=True, tree_name='dataframe_tree', df=df, plot=True, enforce_success=enforce_success)
        # plt.show()

    current_data = mpe[0].to_json() if enforce_success else mpe[0]
    get_task_tree(current_data, model=model, enforce_success=enforce_success)

    
    