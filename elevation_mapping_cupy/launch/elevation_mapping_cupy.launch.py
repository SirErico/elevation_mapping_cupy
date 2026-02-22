from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, PythonExpression
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from ament_index_python.packages import get_package_share_directory
import launch_ros.actions

def generate_launch_description():
    elevation_mapping_cupy_dir = get_package_share_directory('elevation_mapping_cupy')

    is_hardware_arg = DeclareLaunchArgument(
        'is_hardware',
        default_value='false',
        description='Set to true when running on real hardware (best-effort QoS, no sim time)'
    )

    is_hardware = LaunchConfiguration('is_hardware')
    # use_sim_time is the inverse of is_hardware
    use_sim_time = PythonExpression(["'false' if '", is_hardware, "'.lower() in ('true','1') else 'true'"])

    # Config subfolder: is_hardware=true → 'setups/leo01', is_hardware=false → 'core'
    config_subfolder = PythonExpression([
        "'setups/leo01' if '", is_hardware, "'.lower() in ('true','1') else 'core'"
    ])

    node_params = [
        PathJoinSubstitution([
            elevation_mapping_cupy_dir,
            'config',
            config_subfolder,
            'core_param.yaml'
        ]),
        PathJoinSubstitution([
            elevation_mapping_cupy_dir,
            'config',
            config_subfolder,
            'example_setup.yaml'
        ]),
        {'is_hardware': is_hardware}
    ]

    # Simulation: with TF remappings
    sim_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node.py',
        name='elevation_mapping',
        condition=UnlessCondition(is_hardware),
        parameters=node_params,
        remappings=[
            ('/tf', '/j100_0000/tf'),
            ('/tf_static', '/j100_0000/tf_static')],
        output='screen'
    )

    # Hardware: no remappings
    hw_node = Node(
        package='elevation_mapping_cupy',
        executable='elevation_mapping_node.py',
        name='elevation_mapping',
        condition=IfCondition(is_hardware),
        parameters=node_params,
        output='screen'
    )

    return LaunchDescription([
        is_hardware_arg,
        launch_ros.actions.SetParameter(name='use_sim_time', value=use_sim_time),
        sim_node,
        hw_node,
    ]) 