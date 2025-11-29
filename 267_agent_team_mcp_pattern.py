"""
Pattern 267: Agent Team MCP Pattern

This pattern demonstrates team-based agent organization with roles,
responsibilities, and coordinated teamwork.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class AgentTeamState(TypedDict):
    """State for agent team workflow"""
    messages: Annotated[List[str], add]
    project: Dict[str, Any]
    team: List[Dict[str, Any]]
    work_distribution: List[Dict[str, Any]]
    team_progress: Dict[str, Any]


class TeamMember:
    """Individual team member"""
    
    def __init__(self, member_id: str, role: str, skills: List[str], productivity: float):
        self.member_id = member_id
        self.role = role
        self.skills = skills
        self.productivity = productivity
        self.tasks = []
    
    def work_on_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Work on assigned task"""
        task_skill = task.get("required_skill", "")
        skill_match = 1.0 if task_skill in self.skills else 0.5
        
        effort = task.get("effort", 1)
        completion_time = effort / (self.productivity * skill_match)
        quality = 0.95 * skill_match
        
        return {
            "task_id": task.get("id"),
            "member_id": self.member_id,
            "role": self.role,
            "completion_time": completion_time,
            "quality": quality,
            "skill_match": skill_match
        }


class AgentTeam:
    """Manages a team of agents"""
    
    def __init__(self, team_name: str):
        self.team_name = team_name
        self.members = []
        self.team_lead = None
    
    def add_member(self, member: TeamMember, is_lead: bool = False):
        """Add member to team"""
        self.members.append(member)
        if is_lead:
            self.team_lead = member
    
    def distribute_work(self, project_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Distribute work among team members"""
        distribution = []
        
        for task in project_tasks:
            # Find best member for task
            best_member = None
            best_score = 0
            
            for member in self.members:
                required_skill = task.get("required_skill", "")
                if required_skill in member.skills:
                    score = member.productivity
                    if score > best_score:
                        best_score = score
                        best_member = member
            
            if best_member:
                best_member.tasks.append(task)
                distribution.append({
                    "task_id": task.get("id"),
                    "task_name": task.get("name"),
                    "assigned_to": best_member.member_id,
                    "role": best_member.role,
                    "required_skill": task.get("required_skill"),
                    "effort": task.get("effort")
                })
        
        return distribution
    
    def execute_sprint(self) -> Dict[str, Any]:
        """Execute team sprint"""
        all_results = []
        total_time = 0
        total_quality = 0
        
        for member in self.members:
            for task in member.tasks:
                result = member.work_on_task(task)
                all_results.append(result)
                total_time += result["completion_time"]
                total_quality += result["quality"]
        
        avg_quality = total_quality / len(all_results) if all_results else 0
        
        return {
            "tasks_completed": len(all_results),
            "total_time": total_time,
            "average_quality": avg_quality,
            "results": all_results
        }


def form_team_agent(state: AgentTeamState) -> AgentTeamState:
    """Form agent team"""
    print("\n👥 Forming Agent Team...")
    
    project = {
        "name": "Build E-Commerce Platform",
        "duration": "3 months",
        "scope": "Full-stack web application"
    }
    
    team_members = [
        {"id": "Alice", "role": "Tech Lead", "skills": ["architecture", "backend", "frontend"], "productivity": 1.2},
        {"id": "Bob", "role": "Backend Developer", "skills": ["backend", "database", "api"], "productivity": 1.0},
        {"id": "Carol", "role": "Frontend Developer", "skills": ["frontend", "ui", "testing"], "productivity": 1.1},
        {"id": "David", "role": "DevOps Engineer", "skills": ["deployment", "monitoring", "infrastructure"], "productivity": 0.9},
        {"id": "Eve", "role": "QA Engineer", "skills": ["testing", "automation", "quality"], "productivity": 1.0}
    ]
    
    print(f"\n  Project: {project['name']}")
    print(f"  Team Size: {len(team_members)}")
    print(f"\n  Team Members:")
    for member in team_members:
        lead_marker = " (LEAD)" if member["role"] == "Tech Lead" else ""
        print(f"    • {member['id']} - {member['role']}{lead_marker}")
        print(f"      Skills: {', '.join(member['skills'])}")
        print(f"      Productivity: {member['productivity']:.1f}x")
    
    return {
        **state,
        "project": project,
        "team": team_members,
        "messages": [f"✓ Team formed with {len(team_members)} members"]
    }


def distribute_work_agent(state: AgentTeamState) -> AgentTeamState:
    """Distribute work to team"""
    print("\n📋 Distributing Work...")
    
    # Create team
    team = AgentTeam(state["project"]["name"])
    
    for member_data in state["team"]:
        member = TeamMember(
            member_data["id"],
            member_data["role"],
            member_data["skills"],
            member_data["productivity"]
        )
        team.add_member(member, is_lead=(member_data["role"] == "Tech Lead"))
    
    # Define project tasks
    tasks = [
        {"id": "T1", "name": "Design system architecture", "required_skill": "architecture", "effort": 3},
        {"id": "T2", "name": "Implement user API", "required_skill": "backend", "effort": 5},
        {"id": "T3", "name": "Build shopping cart UI", "required_skill": "frontend", "effort": 4},
        {"id": "T4", "name": "Setup CI/CD pipeline", "required_skill": "deployment", "effort": 3},
        {"id": "T5", "name": "Write integration tests", "required_skill": "testing", "effort": 4},
        {"id": "T6", "name": "Implement payment API", "required_skill": "backend", "effort": 5},
        {"id": "T7", "name": "Design responsive UI", "required_skill": "ui", "effort": 4},
        {"id": "T8", "name": "Setup monitoring", "required_skill": "monitoring", "effort": 2}
    ]
    
    # Distribute work
    distribution = team.distribute_work(tasks)
    
    print(f"\n  Total Tasks: {len(tasks)}")
    print(f"\n  Work Distribution:")
    for assignment in distribution:
        print(f"    • {assignment['task_name']}")
        print(f"      Assigned to: {assignment['assigned_to']} ({assignment['role']})")
        print(f"      Skill Required: {assignment['required_skill']}")
        print(f"      Effort: {assignment['effort']} days")
    
    # Execute sprint
    progress = team.execute_sprint()
    
    print(f"\n  Sprint Execution:")
    print(f"    Tasks Completed: {progress['tasks_completed']}")
    print(f"    Total Time: {progress['total_time']:.1f} days")
    print(f"    Average Quality: {progress['average_quality']:.0%}")
    
    return {
        **state,
        "work_distribution": distribution,
        "team_progress": progress,
        "messages": [f"✓ {len(distribution)} tasks distributed and executed"]
    }


def generate_team_report_agent(state: AgentTeamState) -> AgentTeamState:
    """Generate team report"""
    print("\n" + "="*70)
    print("AGENT TEAM REPORT")
    print("="*70)
    
    print(f"\n📁 Project:")
    print(f"  Name: {state['project']['name']}")
    print(f"  Duration: {state['project']['duration']}")
    print(f"  Scope: {state['project']['scope']}")
    
    print(f"\n👥 Team Composition:")
    for member in state["team"]:
        print(f"\n  {member['id']} - {member['role']}")
        print(f"    Skills: {', '.join(member['skills'])}")
        print(f"    Productivity: {member['productivity']:.1f}x")
    
    print(f"\n📋 Work Distribution:")
    # Group by team member
    by_member = {}
    for assignment in state["work_distribution"]:
        member_id = assignment["assigned_to"]
        by_member[member_id] = by_member.get(member_id, []) + [assignment]
    
    for member_id, tasks in by_member.items():
        total_effort = sum(t["effort"] for t in tasks)
        print(f"\n  {member_id}: {len(tasks)} task(s), {total_effort} days")
        for task in tasks:
            print(f"    • {task['task_name']} ({task['effort']} days)")
    
    print(f"\n✅ Sprint Results:")
    progress = state["team_progress"]
    print(f"  Tasks Completed: {progress['tasks_completed']}")
    print(f"  Total Time: {progress['total_time']:.1f} days")
    print(f"  Average Quality: {progress['average_quality']:.0%}")
    
    print(f"\n📊 Individual Performance:")
    for result in progress["results"]:
        print(f"\n  {result['member_id']} ({result['role']}):")
        print(f"    Task: {result['task_id']}")
        print(f"    Time: {result['completion_time']:.1f} days")
        print(f"    Quality: {result['quality']:.0%}")
        print(f"    Skill Match: {result['skill_match']:.0%}")
    
    print(f"\n💡 Agent Team Benefits:")
    print("  • Clear roles and responsibilities")
    print("  • Skill-based task assignment")
    print("  • Coordinated teamwork")
    print("  • Shared goals and objectives")
    print("  • Efficient collaboration")
    print("  • Collective accountability")
    
    # Team metrics
    print(f"\n📈 Team Metrics:")
    if state["team_progress"]["results"]:
        results = state["team_progress"]["results"]
        high_quality = [r for r in results if r["quality"] >= 0.9]
        perfect_match = [r for r in results if r["skill_match"] == 1.0]
        
        print(f"  High Quality Tasks (≥90%): {len(high_quality)}/{len(results)}")
        print(f"  Perfect Skill Matches: {len(perfect_match)}/{len(results)}")
        print(f"  Team Efficiency: {progress['tasks_completed']/progress['total_time']:.2f} tasks/day")
    
    print("\n="*70)
    print("✅ Agent Team Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_agent_team_graph():
    workflow = StateGraph(AgentTeamState)
    workflow.add_node("form", form_team_agent)
    workflow.add_node("distribute", distribute_work_agent)
    workflow.add_node("report", generate_team_report_agent)
    workflow.add_edge(START, "form")
    workflow.add_edge("form", "distribute")
    workflow.add_edge("distribute", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 267: Agent Team MCP Pattern")
    print("="*70)
    
    app = create_agent_team_graph()
    final_state = app.invoke({
        "messages": [],
        "project": {},
        "team": [],
        "work_distribution": [],
        "team_progress": {}
    })
    print("\n✅ Agent Team Pattern Complete!")


if __name__ == "__main__":
    main()
