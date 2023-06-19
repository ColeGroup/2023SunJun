import pandas as pd
import numpy as np
import datetime



def data_process_by_time(time_df, app_protocol, dst_ipv4, dst_port):
    """
    处理2秒内数据
    :param app_protocol: 网络服务类型
    :param dst_ipv4: 目标地址
    :param dst_port: 目标端口
    :param time_df: 2秒内数据dataframe
    :return: 数据集
    """
    # 2s内相同主机 dataframe
    same_main_engine_df = time_df.loc[(time_df['dst_ipv4'] == dst_ipv4) & (time_df['dst_port'] == dst_port)]
    # 2s内相同服务 dataframe
    same_server_df = time_df.loc[time_df['app_protocol'] == app_protocol]
    t_same_main_engine = same_main_engine_df.shape[0]  # 获取2s内相同目标主机
    t_same_server = same_server_df.shape[0]  # 获取2s内相同服务
    # 2s内相同主机中相同服务的占比
    t_same_server_rate = \
        round(
            same_main_engine_df.loc[same_main_engine_df['app_protocol'] == app_protocol].shape[
                0] / t_same_main_engine,
            2)
    # 2s内相同主机中不同服务的占比
    t_diff_server_rate = 1 - t_same_server_rate
    # 2s内相同服务中不同主机的占比
    t_diff_main_engine_rate = \
        round(1 - same_server_df[(same_server_df['dst_ipv4'] == dst_ipv4)
                                 & (same_server_df['dst_port'] == dst_port)].shape[0] / t_same_server, 2)
    data = [t_same_main_engine, t_same_server, t_same_server_rate, t_diff_server_rate, t_diff_main_engine_rate]
    return data

def data_process_by_num(num_df, app_protocol, dst_ipv4, dst_port, src_port):
    """
    处理前100条数据
    :param app_protocol: 网络服务类型
    :param dst_ipv4: 目标地址
    :param dst_port: 目标端口
    :param src_port: 源端口
    :param num_df: 前100条数据dataframe
    :return: 数据集
    """
    # 前100个连接中，相同目标主机 dataframe
    same_main_engine_data = num_df.loc[(num_df['dst_ipv4'] == dst_ipv4) & (num_df['dst_port'] == dst_port)]
    same_main_engine = same_main_engine_data.shape[0]  # 前100个连接中，相同目标主机的数量
    # 前100个连接中，相同目标主机中相同服务的数量
    same_server = same_main_engine_data.loc[same_main_engine_data['app_protocol'] == app_protocol].shape[0]
    same_server_rate =  0 if same_main_engine<=0 else round(same_server / same_main_engine, 2)  # 前100个连接中，相同目标主机中相同服务的占比
    diff_server_rate = round(1 - same_server_rate, 2)  # 前100个连接中，相同目标中不同服务的占比
    # 前100个连接中，相同目标中相同源端口的占比
    same_src_rate = \
        0 if same_main_engine<=0 else round(same_main_engine_data.loc[same_main_engine_data['src_port'] == src_port].shape[0] / same_main_engine,
              2)
    diff_src_rate = round(1 - same_src_rate, 2)  # 前100个连接中，相同目标中不同源端口的占比
    data = [same_main_engine, same_server, same_server_rate, diff_server_rate, same_src_rate, diff_src_rate]
    return data

def data_std_mean_num(num_df):
    """
    统计前n条数据的均值和标准差
    :param num_df: 数据dataframe
    :return: 数据集
    """
    send_col = num_df['send_bytes'].values.astype(float)
    mean = np.mean(send_col)
    std = np.std(send_col)
    # mean= send_col.mean()
    # std = send_col.std()
    data = [std, mean]
    return data

def data_package_rate(num_df):
    """
    统计前n条数据的各个类型包占比
    :param num_df: 数据dataframe
    :return: 数据集
    """
    rst_col = num_df['rst_flag'].values.astype(int)
    rst_rate = round(np.sum(rst_col >0 )/rst_col.shape[0],5)
    data = [rst_rate]
    return data


def data_pree(file):
    or_data = pd.read_csv("static/%s"%(file))
    df = or_data.sort_values(by='session_start_time')  # 按 session_start_time 升序排列
    df.reset_index(inplace=True)  # 重置索引
    start_time = df['session_start_time'].apply(int)
    result = []
    for ind, row in df.iterrows():
        if ind < 1000:
            continue
        session_start_time = int(row['session_start_time'])  # 当前行结束时间
        session_end_time = int(row['session_end_time'])  # 当前行结束时间
        duration = session_end_time - session_start_time  # 连接时长
        if duration < 0:
            continue
        trans_protocol = row['trans_prototol']  # 当前协议类型
        app_protocol = row['app_protocol']  # 当前网络服务类型
        send_bytes = row['send_bytes']  # 获取源到目标字节数
        recv_bytes = row['recv_bytes']  # 获取目标到源字节数
        dst_ipv4 = row["dst_ipv4"]  # 获取目标地址
        dst_port = row["dst_port"]  # 获取目标端口
        src_port = row["src_port"]  # 获取源端口
        # src_ipv4 = row['src_ipv4']
        data = [dst_ipv4,duration, trans_protocol, app_protocol, send_bytes, recv_bytes]
        time_df = df.loc[(start_time <= session_start_time) & (start_time >= session_start_time - 2000000)]
        # 获得前1s内的数据 yxd
        time_df_2 = df.loc[(start_time <= session_start_time) & (start_time >= session_start_time - 1000000)]
        num_df = df[ind - 100: ind + 1]  # 获取当前行前100条数据
        # 获取前1000条数据 yxd
        num_df_1000 = df[ind - 1000: ind + 1]
        time_data = data_process_by_time(time_df, app_protocol, dst_ipv4, dst_port)
        # 获得前1s内数据统计值 yxd
        time_data_2 = data_process_by_time(time_df_2, app_protocol, dst_ipv4, dst_port)
        num_data = data_process_by_num(num_df, app_protocol, dst_ipv4, dst_port, src_port)
        # 获取前1000条数据的均值和标准差 yxd
        num_data_std_mean = data_std_mean_num(num_df_1000)
        num_data_package_rate = data_package_rate(num_df_1000)
        data.extend(time_data)
        # 1s yxd
        data.extend(time_data_2)
        data.extend(num_data)
        data.extend(num_data_std_mean)
        data.extend(num_data_package_rate)
        result.append(data)
    return result


