# src/utils/config.py

"""
Configuration module for the little-matrix simulation.

This module defines the `Config` class and the `load_config` function,
which loads configuration settings from a YAML file into structured data classes.
It provides a centralized way to manage simulation parameters and settings.

Classes:
    Config: Main configuration class encompassing all settings.

Functions:
    load_config(config_file: str) -> Config: Loads configuration settings from a YAML file.
"""

import yaml
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

# Define simulation configuration dataclasses
@dataclass
class SimulationConfig:
    name: str
    description: str
    render: bool
    timesteps: int
    save_state_interval: Optional[int] = None
    llm_metrics_interval: int = 50
    seed: Optional[int] = None

@dataclass
class TerrainTypeConfig:
    name: str
    symbol: str
    movement_cost: int
    feature_type: str
    is_impassable: bool = False
    color: List[int] = field(default_factory=list)

@dataclass
class TerrainConfig:
    types: List[TerrainTypeConfig]
    distribution: Dict[str, int]
    wrap_around: bool = False

@dataclass
class GridConfig:
    width: int
    height: int
    wrap_around: bool = False
    terrain: Optional[TerrainConfig] = None

@dataclass
class ResourceTypeConfig:
    name: str
    symbol: str
    color: List[int]
    spawn_on: List[str]
    quantity_range: List[int]

@dataclass
class ResourceConfig:
    spawn_rate: float
    max_resources: int
    regeneration_rate: float
    types: List[ResourceTypeConfig]

@dataclass
class WeatherEffectConfig:
    type: str
    value: Union[float, int]

@dataclass
class WeatherPatternConfig:
    name: str
    duration_range: List[int]
    effects: List[WeatherEffectConfig]

@dataclass
class WeatherTransitionConfig:
    from_pattern: str
    to_pattern: str
    probability: float

@dataclass
class WeatherConfig:
    enabled: bool
    change_interval: int
    patterns: List[WeatherPatternConfig]
    transitions: List[WeatherTransitionConfig]

@dataclass
class EnvironmentConfig:
    grid: GridConfig
    resource: ResourceConfig
    weather: Optional[WeatherConfig] = None

@dataclass
class AgentTypeBehaviorTraits:
    resource_preference: Optional[str] = None
    risk_tolerance: Optional[float] = None
    speed_modifier: Optional[float] = None
    intelligence: Optional[float] = None
    gathering_efficiency: Optional[float] = None
    storage_capacity: Optional[float] = None
    protective_instinct: Optional[bool] = None
    combat_skill: Optional[float] = None
    armor: Optional[float] = None

@dataclass
class AgentTypeConfig:
    name: str
    color: List[int]
    symbol: str
    behavior_traits: AgentTypeBehaviorTraits

@dataclass
class StatusEffectConfig:
    name: str
    duration: List[int]
    health_decrease_per_tick: Optional[int] = None
    energy_recovery_modifier: Optional[float] = None
    action_success_modifier: Optional[float] = None

@dataclass
class AgentsCustomizationConfig:
    types: List[AgentTypeConfig]
    status_effects_enabled: bool = False
    status_effects: List[StatusEffectConfig] = field(default_factory=list)

@dataclass
class SocialConfig:
    friendships: Dict[str, Any]
    rivalries: Dict[str, Any]

@dataclass
class LearningMethodConfig:
    name: str
    parameters: Dict[str, Any]

@dataclass
class LearningConfig:
    enabled: bool
    methods: List[LearningMethodConfig]

@dataclass
class AgentsBehaviorConfig:
    initial_energy: int
    communication_enabled: bool
    aggressiveness_base: float
    aggressiveness_modifiers: Dict[str, float]
    energy_consumption_rate: float

@dataclass
class AgentsPerceptionConfig:
    base_radius: int
    modifiers: Dict[str, int]
    base_range: int
    sight_modifiers: Dict[str, int]

@dataclass
class AgentsConfig:
    count: int
    perception: AgentsPerceptionConfig
    behavior: AgentsBehaviorConfig
    customization: AgentsCustomizationConfig
    social: Optional[SocialConfig] = None
    learning: Optional[LearningConfig] = None

@dataclass
class RendererDisplayConfig:
    size: List[int]
    fps: int
    fullscreen: bool = False
    resizable: bool = False

@dataclass
class RendererColorsConfig:
    background: List[int]
    agent_default: List[int]
    resource: List[int]
    terrain: Dict[str, List[int]]
    environment_effects: Dict[str, List[int]]

@dataclass
class RendererEffectsConfig:
    show_shadows: bool
    particle_effects: bool
    weather_effects: bool
    smooth_movement: bool
    display_hud: bool
    show_agent_names: bool
    show_health_bars: bool
    enable_antialiasing: bool = False
    show_grid: bool = False
    highlight_selected_agent: bool = False
@dataclass
class RendererConfig:
    display: RendererDisplayConfig
    colors: RendererColorsConfig
    effects: RendererEffectsConfig

@dataclass
class LoggingRotationConfig:
    enabled: bool
    max_bytes: int
    backup_count: int

@dataclass
class LoggingConfig:
    level: str
    format: str
    file: Optional[str] = None
    rotation: Optional[LoggingRotationConfig] = None

@dataclass
class PerformanceConfig:
    max_agent_threads: int
    tick_rate: float
    use_multiprocessing: bool = False
    max_db_connections: Optional[int] = 5

@dataclass
class ExperimentModeConfig:
    enabled: bool
    data_collection: bool
    save_interval: int
    data_fields: List[str]

@dataclass
class ProfilingConfig:
    enabled: bool
    output_file: str

@dataclass
class AIDebuggingConfig:
    enabled: bool
    log_decisions: bool
    visualize_decision_trees: bool
    profiling: ProfilingConfig

@dataclass
class CompatibilityModeConfig:
    enabled: bool
    reduce_graphics: bool
    limit_agents: int

@dataclass
class AdvancedConfig:
    performance: PerformanceConfig
    experiment_mode: Optional[ExperimentModeConfig] = None
    ai_debugging: Optional[AIDebuggingConfig] = None
    compatibility_mode: Optional[CompatibilityModeConfig] = None
    database_file: Optional[str] = "little_matrix.db"

@dataclass
class LLMConfig:
    api_key: str
    model: str

@dataclass
class Config:
    simulation: SimulationConfig
    environment: EnvironmentConfig
    agents: AgentsConfig
    renderer: RendererConfig
    logging: LoggingConfig
    advanced: AdvancedConfig
    llm: Optional[LLMConfig] = None

def load_config(config_file: str) -> Config:
    """
    Loads configuration settings from a YAML file and returns a Config object.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        Config: Configuration settings as a Config object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    config_path = Path(config_file)
    if not config_path.is_file():
        logger.error(f"Configuration file '{config_file}' not found.")
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    
    try:
        with open(config_file, 'r') as file:
            config_dict = yaml.safe_load(file)
            logger.info(f"Configuration loaded from '{config_file}'.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{config_file}': {e}")
        raise

    try:
        # Load LLM config only if both required keys are present
        llm_data = config_dict.get('llm')
        llm_config = LLMConfig(api_key=llm_data['api_key'], model=llm_data['model']) if llm_data and 'api_key' in llm_data and 'model' in llm_data else None

        # Build the Config object from YAML data
        config = Config(
            simulation=SimulationConfig(**config_dict['simulation']),
            environment=EnvironmentConfig(
                grid=GridConfig(
                    width=config_dict['environment']['grid']['width'],
                    height=config_dict['environment']['grid']['height'],
                    wrap_around=config_dict['environment']['grid'].get('wrap_around', False),
                    terrain=TerrainConfig(
                        types=[TerrainTypeConfig(**tt) for tt in config_dict['environment']['grid']['terrain']['types']],
                        distribution=config_dict['environment']['grid']['terrain']['distribution']
                    )
                ),
                resource=ResourceConfig(
                    spawn_rate=config_dict['environment']['resource']['spawn_rate'],
                    max_resources=config_dict['environment']['resource']['max_resources'],
                    regeneration_rate=config_dict['environment']['resource']['regeneration_rate'],
                    types=[ResourceTypeConfig(**rt) for rt in config_dict['environment']['resource']['types']]
                ),
                weather=WeatherConfig(
                    enabled=config_dict['environment']['weather']['enabled'],
                    change_interval=config_dict['environment']['weather']['change_interval'],
                    patterns=[WeatherPatternConfig(**wp) for wp in config_dict['environment']['weather']['patterns']],
                    transitions=[
                        WeatherTransitionConfig(from_pattern=fp, to_pattern=tp, probability=p)
                        for fp, tp_dict in config_dict['environment']['weather']['transitions'].items()
                        for tp, p in tp_dict.items()
                    ]
                )
            ),
            agents=AgentsConfig(
                count=config_dict['agents']['count'],
                perception=AgentsPerceptionConfig(
                    base_radius=config_dict['agents']['perception']['base_radius'],
                    modifiers=config_dict['agents']['perception']['modifiers'],
                    base_range=config_dict['agents']['sight_range']['base_range'],
                    sight_modifiers=config_dict['agents']['sight_range']['modifiers']
                ),
                behavior=AgentsBehaviorConfig(
                    initial_energy=config_dict['agents']['behavior']['initial_energy'],
                    communication_enabled=config_dict['agents']['behavior']['communication_enabled'],
                    aggressiveness_base=config_dict['agents']['behavior']['aggressiveness_base'],
                    aggressiveness_modifiers=config_dict['agents']['behavior']['aggressiveness_modifiers'],
                    energy_consumption_rate=config_dict['agents']['behavior']['energy_consumption_rate']
                ),
                customization=AgentsCustomizationConfig(
                    types=[AgentTypeConfig(**at) for at in config_dict['agents']['customization']['types']],
                    status_effects_enabled=config_dict['agents']['customization']['status_effects']['enabled'],
                    status_effects=[StatusEffectConfig(**se) for se in config_dict['agents']['customization']['status_effects']['effects']]
                ),
                social=SocialConfig(
                    friendships=config_dict['agents']['social']['friendships'],
                    rivalries=config_dict['agents']['social']['rivalries']
                ) if 'social' in config_dict['agents'] else None,
                learning=LearningConfig(
                    enabled=config_dict['agents']['learning']['enabled'],
                    methods=[LearningMethodConfig(**lm) for lm in config_dict['agents']['learning']['methods']]
                ) if 'learning' in config_dict['agents'] else None
            ),
            renderer=RendererConfig(
                display=RendererDisplayConfig(**config_dict['renderer']['display']),
                colors=RendererColorsConfig(
                    background=config_dict['renderer']['colors']['background'],
                    agent_default=config_dict['renderer']['colors']['agent_default'],
                    resource=config_dict['renderer']['colors']['resource'],
                    terrain=config_dict['renderer']['colors']['terrain'],
                    environment_effects=config_dict['renderer']['colors']['environment_effects']
                ),
                effects=RendererEffectsConfig(**config_dict['renderer']['effects'])
            ),
            logging=LoggingConfig(
                level=config_dict['logging']['level'],
                format=config_dict['logging']['format'],
                file=config_dict['logging'].get('file'),
                rotation=LoggingRotationConfig(
                    enabled=config_dict['logging']['rotation']['enabled'],
                    max_bytes=config_dict['logging']['rotation']['max_bytes'],
                    backup_count=config_dict['logging']['rotation']['backup_count']
                ) if 'rotation' in config_dict['logging'] else None
            ),
            advanced=AdvancedConfig(
                performance=PerformanceConfig(**config_dict['advanced']['performance']),
                experiment_mode=ExperimentModeConfig(
                    enabled=config_dict['advanced']['experiment_mode']['enabled'],
                    data_collection=config_dict['advanced']['experiment_mode']['data_collection'],
                    save_interval=config_dict['advanced']['experiment_mode']['save_interval'],
                    data_fields=config_dict['advanced']['experiment_mode']['data_fields']
                ) if 'experiment_mode' in config_dict['advanced'] else None,
                ai_debugging=AIDebuggingConfig(
                    enabled=config_dict['advanced']['ai_debugging']['enabled'],
                    log_decisions=config_dict['advanced']['ai_debugging']['log_decisions'],
                    visualize_decision_trees=config_dict['advanced']['ai_debugging']['visualize_decision_trees'],
                    profiling=ProfilingConfig(
                        enabled=config_dict['advanced']['ai_debugging']['profiling']['enabled'],
                        output_file=config_dict['advanced']['ai_debugging']['profiling']['output_file']
                    )
                ) if 'ai_debugging' in config_dict['advanced'] else None,
                compatibility_mode=CompatibilityModeConfig(**config_dict['advanced']['compatibility_mode']) if 'compatibility_mode' in config_dict['advanced'] else None
            ),
            llm=llm_config
        )
        logger.info("Configuration successfully parsed into Config object.")
        return config

    except KeyError as e:
        logger.error(f"Missing configuration key: {e}")
        raise ValueError(f"Missing configuration key: {e}")
    except TypeError as e:
        logger.error(f"Incorrect configuration type: {e}")
        raise TypeError(f"Incorrect configuration type: {e}")
