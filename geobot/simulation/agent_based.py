"""
Agent-Based Modeling for Geopolitical Simulation

Models individual actors (states, organizations, leaders) and their interactions.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AgentType(Enum):
    """Types of geopolitical agents."""
    STATE = "state"
    ORGANIZATION = "organization"
    LEADER = "leader"
    ALLIANCE = "alliance"


@dataclass
class AgentState:
    """State variables for an agent."""
    position: np.ndarray  # Position in feature space
    resources: float = 1.0
    power: float = 1.0
    hostility: float = 0.0
    cooperation: float = 0.5
    stability: float = 1.0
    custom: Dict[str, float] = field(default_factory=dict)


class GeopoliticalAgent:
    """
    Represents a geopolitical actor.

    Agents have internal state, decision-making logic, and
    interact with other agents and the environment.
    """

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        initial_state: AgentState
    ):
        """
        Initialize agent.

        Parameters
        ----------
        agent_id : str
            Unique agent identifier
        agent_type : AgentType
            Type of agent
        initial_state : AgentState
            Initial state
        """
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.state = initial_state
        self.history: List[AgentState] = [initial_state]
        self.relationships: Dict[str, float] = {}  # {agent_id: relationship_strength}

    def update_state(self, **kwargs) -> None:
        """Update agent state variables."""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
            else:
                self.state.custom[key] = value

    def decide_action(
        self,
        environment: Dict[str, Any],
        other_agents: List['GeopoliticalAgent']
    ) -> Dict[str, Any]:
        """
        Decide on action based on current state and environment.

        Parameters
        ----------
        environment : dict
            Environmental factors
        other_agents : list
            Other agents in the system

        Returns
        -------
        dict
            Chosen action
        """
        # Simple decision logic (can be made more sophisticated)
        action = {
            'type': 'none',
            'target': None,
            'intensity': 0.0
        }

        # Check for threats
        threats = [
            agent for agent in other_agents
            if self.relationships.get(agent.agent_id, 0) < -0.5
            and agent.state.power > self.state.power * 0.8
        ]

        if threats and self.state.hostility > 0.5:
            # Consider conflict
            action = {
                'type': 'escalate',
                'target': threats[0].agent_id,
                'intensity': self.state.hostility
            }
        elif self.state.cooperation > 0.7:
            # Seek cooperation
            potential_partners = [
                agent for agent in other_agents
                if self.relationships.get(agent.agent_id, 0) > 0.3
            ]
            if potential_partners:
                action = {
                    'type': 'cooperate',
                    'target': potential_partners[0].agent_id,
                    'intensity': self.state.cooperation
                }

        return action

    def interact(self, other_agent: 'GeopoliticalAgent', action: Dict[str, Any]) -> None:
        """
        Interact with another agent.

        Parameters
        ----------
        other_agent : GeopoliticalAgent
            Other agent
        action : dict
            Action to perform
        """
        action_type = action['type']
        intensity = action['intensity']

        if action_type == 'cooperate':
            # Strengthen relationship
            current_rel = self.relationships.get(other_agent.agent_id, 0)
            self.relationships[other_agent.agent_id] = min(1.0, current_rel + 0.1 * intensity)

            # Mutual benefit
            self.state.resources += 0.05 * intensity
            other_agent.state.resources += 0.05 * intensity

        elif action_type == 'escalate':
            # Weaken relationship
            current_rel = self.relationships.get(other_agent.agent_id, 0)
            self.relationships[other_agent.agent_id] = max(-1.0, current_rel - 0.2 * intensity)

            # Conflict effects
            power_ratio = self.state.power / (other_agent.state.power + 1e-6)
            if power_ratio > 1:
                self.state.resources += 0.1 * intensity
                other_agent.state.resources -= 0.15 * intensity
                other_agent.state.stability -= 0.1 * intensity
            else:
                self.state.resources -= 0.1 * intensity
                self.state.stability -= 0.05 * intensity

    def save_state(self) -> None:
        """Save current state to history."""
        self.history.append(AgentState(
            position=self.state.position.copy(),
            resources=self.state.resources,
            power=self.state.power,
            hostility=self.state.hostility,
            cooperation=self.state.cooperation,
            stability=self.state.stability,
            custom=self.state.custom.copy()
        ))


class AgentBasedModel:
    """
    Agent-based model for geopolitical simulation.

    Manages multiple agents and their interactions over time.
    """

    def __init__(self):
        """Initialize agent-based model."""
        self.agents: Dict[str, GeopoliticalAgent] = {}
        self.environment: Dict[str, Any] = {}
        self.time: int = 0

    def add_agent(self, agent: GeopoliticalAgent) -> None:
        """
        Add agent to model.

        Parameters
        ----------
        agent : GeopoliticalAgent
            Agent to add
        """
        self.agents[agent.agent_id] = agent

    def remove_agent(self, agent_id: str) -> None:
        """
        Remove agent from model.

        Parameters
        ----------
        agent_id : str
            Agent ID to remove
        """
        if agent_id in self.agents:
            del self.agents[agent_id]

    def set_environment(self, **kwargs) -> None:
        """Set environmental variables."""
        self.environment.update(kwargs)

    def step(self) -> None:
        """
        Execute one time step of simulation.

        All agents make decisions and interact.
        """
        self.time += 1

        # Phase 1: All agents decide actions
        actions = {}
        other_agents_list = list(self.agents.values())

        for agent in self.agents.values():
            action = agent.decide_action(self.environment, other_agents_list)
            actions[agent.agent_id] = action

        # Phase 2: Execute actions
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]

            if action['type'] != 'none' and action['target'] is not None:
                if action['target'] in self.agents:
                    target = self.agents[action['target']]
                    agent.interact(target, action)

        # Phase 3: Environmental updates
        for agent in self.agents.values():
            # Resource growth
            agent.state.resources *= (1 + 0.01 * agent.state.stability)

            # Power calculation
            agent.state.power = agent.state.resources * agent.state.stability

            # Add noise
            agent.state.hostility += np.random.normal(0, 0.05)
            agent.state.hostility = np.clip(agent.state.hostility, 0, 1)

            agent.state.cooperation += np.random.normal(0, 0.05)
            agent.state.cooperation = np.clip(agent.state.cooperation, 0, 1)

            # Save state
            agent.save_state()

    def run(self, n_steps: int) -> None:
        """
        Run simulation for multiple steps.

        Parameters
        ----------
        n_steps : int
            Number of time steps
        """
        for _ in range(n_steps):
            self.step()

    def get_agent_trajectories(self, agent_id: str) -> Dict[str, List[float]]:
        """
        Get historical trajectories for an agent.

        Parameters
        ----------
        agent_id : str
            Agent ID

        Returns
        -------
        dict
            Trajectories of state variables
        """
        if agent_id not in self.agents:
            return {}

        agent = self.agents[agent_id]
        history = agent.history

        return {
            'resources': [s.resources for s in history],
            'power': [s.power for s in history],
            'hostility': [s.hostility for s in history],
            'cooperation': [s.cooperation for s in history],
            'stability': [s.stability for s in history]
        }

    def get_system_state(self) -> Dict[str, Any]:
        """
        Get current state of entire system.

        Returns
        -------
        dict
            System state
        """
        return {
            'time': self.time,
            'n_agents': len(self.agents),
            'total_resources': sum(a.state.resources for a in self.agents.values()),
            'mean_hostility': np.mean([a.state.hostility for a in self.agents.values()]),
            'mean_cooperation': np.mean([a.state.cooperation for a in self.agents.values()]),
            'mean_stability': np.mean([a.state.stability for a in self.agents.values()])
        }

    def analyze_network(self) -> Dict[str, Any]:
        """
        Analyze the network of relationships.

        Returns
        -------
        dict
            Network metrics
        """
        import networkx as nx

        # Build network
        G = nx.Graph()
        for agent in self.agents.values():
            G.add_node(agent.agent_id)

        for agent in self.agents.values():
            for other_id, strength in agent.relationships.items():
                if strength > 0.1:  # Only positive relationships
                    G.add_edge(agent.agent_id, other_id, weight=strength)

        # Compute metrics
        return {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'average_clustering': nx.average_clustering(G) if G.number_of_edges() > 0 else 0,
            'connected_components': nx.number_connected_components(G)
        }
