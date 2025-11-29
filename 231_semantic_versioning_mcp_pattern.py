"""
Pattern 231: Semantic Versioning MCP Pattern

This pattern demonstrates semantic versioning (SemVer) - a versioning scheme
that uses MAJOR.MINOR.PATCH format to communicate the nature of changes.
"""

from typing import TypedDict, Annotated, List
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from dataclasses import dataclass


# State definition
class SemanticVersionState(TypedDict):
    """State for semantic versioning workflow"""
    messages: Annotated[List[str], add]
    version_history: List[dict]
    current_version: str
    change_log: List[str]


@dataclass
class SemanticVersion:
    """Semantic version following MAJOR.MINOR.PATCH format"""
    major: int
    minor: int
    patch: int
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def bump_major(self) -> 'SemanticVersion':
        """
        Increment MAJOR version - incompatible API changes
        Resets MINOR and PATCH to 0
        """
        return SemanticVersion(self.major + 1, 0, 0)
    
    def bump_minor(self) -> 'SemanticVersion':
        """
        Increment MINOR version - backwards-compatible new features
        Resets PATCH to 0
        """
        return SemanticVersion(self.major, self.minor + 1, 0)
    
    def bump_patch(self) -> 'SemanticVersion':
        """
        Increment PATCH version - backwards-compatible bug fixes
        """
        return SemanticVersion(self.major, self.minor, self.patch + 1)
    
    def compare(self, other: 'SemanticVersion') -> int:
        """
        Compare versions
        Returns: -1 if self < other, 0 if equal, 1 if self > other
        """
        if self.major != other.major:
            return 1 if self.major > other.major else -1
        if self.minor != other.minor:
            return 1 if self.minor > other.minor else -1
        if self.patch != other.patch:
            return 1 if self.patch > other.patch else -1
        return 0


# Change types
@dataclass
class ChangeRequest:
    """Represents a change request"""
    change_type: str  # 'breaking', 'feature', 'bugfix'
    description: str
    ticket_id: str


# Agent functions
def initialize_version_agent(state: SemanticVersionState) -> SemanticVersionState:
    """Agent that initializes semantic versioning system"""
    print("\n🔢 Initializing Semantic Versioning System...")
    
    initial_version = SemanticVersion(1, 0, 0)
    
    version_history = [
        {
            "version": str(initial_version),
            "date": "2024-01-01",
            "changes": ["Initial release"],
            "type": "MAJOR"
        }
    ]
    
    print(f"\n  Initial Version: {initial_version}")
    print("\n  Semantic Versioning Format: MAJOR.MINOR.PATCH")
    print("    MAJOR: Incompatible API changes")
    print("    MINOR: Backwards-compatible new features")
    print("    PATCH: Backwards-compatible bug fixes")
    
    return {
        **state,
        "current_version": str(initial_version),
        "version_history": version_history,
        "messages": [f"✓ Initialized at version {initial_version}"]
    }


def plan_changes_agent(state: SemanticVersionState) -> SemanticVersionState:
    """Agent that plans version changes based on change requests"""
    print("\n📋 Planning Version Changes...")
    
    # Simulate change requests
    change_requests = [
        ChangeRequest("bugfix", "Fixed user login timeout issue", "BUG-101"),
        ChangeRequest("bugfix", "Corrected date formatting in reports", "BUG-102"),
        ChangeRequest("feature", "Added export to CSV functionality", "FEAT-201"),
        ChangeRequest("feature", "Added dark mode support", "FEAT-202"),
        ChangeRequest("breaking", "Removed deprecated v1 API endpoints", "BREAK-301"),
        ChangeRequest("bugfix", "Fixed memory leak in cache", "BUG-103"),
    ]
    
    print(f"\n  Processing {len(change_requests)} change requests:")
    
    # Categorize changes
    bugfixes = [c for c in change_requests if c.change_type == "bugfix"]
    features = [c for c in change_requests if c.change_type == "feature"]
    breaking = [c for c in change_requests if c.change_type == "breaking"]
    
    print(f"    🐛 Bug Fixes: {len(bugfixes)}")
    for bug in bugfixes:
        print(f"       - {bug.description} ({bug.ticket_id})")
    
    print(f"    ✨ New Features: {len(features)}")
    for feat in features:
        print(f"       - {feat.description} ({feat.ticket_id})")
    
    print(f"    💥 Breaking Changes: {len(breaking)}")
    for brk in breaking:
        print(f"       - {brk.description} ({brk.ticket_id})")
    
    change_log = [f"{c.change_type.upper()}: {c.description}" for c in change_requests]
    
    return {
        **state,
        "change_log": change_log,
        "messages": [f"✓ Planned {len(change_requests)} changes"]
    }


def determine_version_bump_agent(state: SemanticVersionState) -> SemanticVersionState:
    """Agent that determines appropriate version bump"""
    print("\n🎯 Determining Version Bump...")
    
    # Parse current version
    current = SemanticVersion(*map(int, state["current_version"].split('.')))
    
    # Analyze changes to determine bump type
    has_breaking = any("BREAKING:" in change for change in state["change_log"])
    has_feature = any("FEATURE:" in change for change in state["change_log"])
    has_bugfix = any("BUGFIX:" in change for change in state["change_log"])
    
    if has_breaking:
        new_version = current.bump_major()
        bump_type = "MAJOR"
        reason = "Breaking changes detected - incompatible API changes"
    elif has_feature:
        new_version = current.bump_minor()
        bump_type = "MINOR"
        reason = "New features added - backwards-compatible"
    elif has_bugfix:
        new_version = current.bump_patch()
        bump_type = "PATCH"
        reason = "Bug fixes only - backwards-compatible"
    else:
        new_version = current
        bump_type = "NONE"
        reason = "No changes detected"
    
    print(f"\n  Current Version: {current}")
    print(f"  New Version: {new_version}")
    print(f"  Bump Type: {bump_type}")
    print(f"  Reason: {reason}")
    
    return {
        **state,
        "current_version": str(new_version),
        "messages": [f"✓ Version bumped: {current} → {new_version} ({bump_type})"]
    }


def update_version_history_agent(state: SemanticVersionState) -> SemanticVersionState:
    """Agent that updates version history"""
    print("\n📚 Updating Version History...")
    
    new_entry = {
        "version": state["current_version"],
        "date": "2024-01-15",  # Simulated date
        "changes": state["change_log"],
        "type": "MAJOR" if ".0.0" in state["current_version"] and state["current_version"] != "1.0.0" else
                "MINOR" if ".0" in state["current_version"] and not ".0.0" in state["current_version"] else
                "PATCH"
    }
    
    version_history = state["version_history"] + [new_entry]
    
    print(f"\n  Added version {state['current_version']} to history")
    print(f"  Total versions tracked: {len(version_history)}")
    
    return {
        **state,
        "version_history": version_history,
        "messages": [f"✓ History updated with version {state['current_version']}"]
    }


def generate_release_notes_agent(state: SemanticVersionState) -> SemanticVersionState:
    """Agent that generates release notes"""
    print("\n" + "="*70)
    print("SEMANTIC VERSIONING RELEASE NOTES")
    print("="*70)
    
    print(f"\n📦 Version: {state['current_version']}")
    print(f"📅 Release Date: 2024-01-15")
    
    print("\n📋 Changes in This Release:")
    
    # Categorize changes
    breaking_changes = [c for c in state["change_log"] if "BREAKING:" in c]
    new_features = [c for c in state["change_log"] if "FEATURE:" in c]
    bug_fixes = [c for c in state["change_log"] if "BUGFIX:" in c]
    
    if breaking_changes:
        print("\n  💥 BREAKING CHANGES:")
        for i, change in enumerate(breaking_changes, 1):
            print(f"    {i}. {change.replace('BREAKING: ', '')}")
    
    if new_features:
        print("\n  ✨ New Features:")
        for i, change in enumerate(new_features, 1):
            print(f"    {i}. {change.replace('FEATURE: ', '')}")
    
    if bug_fixes:
        print("\n  🐛 Bug Fixes:")
        for i, change in enumerate(bug_fixes, 1):
            print(f"    {i}. {change.replace('BUGFIX: ', '')}")
    
    print("\n📚 Version History:")
    for entry in state["version_history"]:
        print(f"\n  Version {entry['version']} ({entry['type']}) - {entry['date']}")
        print(f"    Changes: {len(entry['changes'])}")
    
    print("\n💡 Semantic Versioning Guidelines:")
    print("  • MAJOR version (X.0.0): Incompatible API changes")
    print("  • MINOR version (1.X.0): Backwards-compatible new features")
    print("  • PATCH version (1.0.X): Backwards-compatible bug fixes")
    print("  • Start at 1.0.0 for first stable release")
    print("  • Use 0.x.x for initial development")
    
    print("\n" + "="*70)
    print(f"✅ Version {state['current_version']} Released!")
    print("="*70)
    
    return {
        **state,
        "messages": ["✓ Release notes generated"]
    }


# Create the graph
def create_semantic_versioning_graph():
    """Create the semantic versioning workflow graph"""
    workflow = StateGraph(SemanticVersionState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_version_agent)
    workflow.add_node("plan_changes", plan_changes_agent)
    workflow.add_node("determine_bump", determine_version_bump_agent)
    workflow.add_node("update_history", update_version_history_agent)
    workflow.add_node("generate_notes", generate_release_notes_agent)
    
    # Add edges
    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "plan_changes")
    workflow.add_edge("plan_changes", "determine_bump")
    workflow.add_edge("determine_bump", "update_history")
    workflow.add_edge("update_history", "generate_notes")
    workflow.add_edge("generate_notes", END)
    
    return workflow.compile()


def main():
    """Main execution function"""
    print("="*70)
    print("Pattern 231: Semantic Versioning MCP Pattern")
    print("="*70)
    print("\nSemantic Versioning: Communicate change impact through version numbers")
    print("Format: MAJOR.MINOR.PATCH")
    
    # Create and run the workflow
    app = create_semantic_versioning_graph()
    
    # Initial state
    initial_state = {
        "messages": [],
        "version_history": [],
        "current_version": "1.0.0",
        "change_log": []
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    print("\n✅ Semantic Versioning Pattern Complete!")


if __name__ == "__main__":
    main()
