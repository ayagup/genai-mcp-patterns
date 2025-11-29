"""
Pattern 270: Agent Auction MCP Pattern

This pattern demonstrates agent auction mechanisms where agents bid
for resources, tasks, or opportunities using various auction protocols.
"""

from typing import TypedDict, Annotated, List, Dict, Any
from operator import add
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


class AgentAuctionState(TypedDict):
    """State for agent auction workflow"""
    messages: Annotated[List[str], add]
    auction_items: List[Dict[str, Any]]
    bidders: List[Dict[str, Any]]
    auction_rounds: List[Dict[str, Any]]
    auction_results: List[Dict[str, Any]]


class BiddingAgent:
    """Agent that participates in auctions"""
    
    def __init__(self, agent_id: str, budget: float, valuation_strategy: str):
        self.agent_id = agent_id
        self.budget = budget
        self.remaining_budget = budget
        self.valuation_strategy = valuation_strategy
        self.won_items = []
    
    def calculate_bid(self, item: Dict[str, Any], current_price: float, round_num: int) -> float:
        """Calculate bid for an item"""
        # Estimate item value based on strategy
        estimated_value = self._estimate_value(item)
        
        # Won't bid above estimated value or budget
        max_bid = min(estimated_value, self.remaining_budget)
        
        if current_price >= max_bid:
            return 0  # Don't bid
        
        # Bidding strategy
        if self.valuation_strategy == "aggressive":
            # Bid close to maximum
            bid = current_price + (max_bid - current_price) * 0.8
        elif self.valuation_strategy == "conservative":
            # Bid cautiously
            bid = current_price + (max_bid - current_price) * 0.2
        else:  # balanced
            # Moderate bidding
            bid = current_price + (max_bid - current_price) * 0.5
        
        # Round to 2 decimal places
        return round(min(bid, self.remaining_budget), 2)
    
    def _estimate_value(self, item: Dict[str, Any]) -> float:
        """Estimate value of item"""
        base_value = item.get("base_value", 100)
        quality = item.get("quality", 0.5)
        utility = item.get("utility", 0.5)
        
        # Different strategies value items differently
        if self.valuation_strategy == "aggressive":
            return base_value * (1 + quality) * (1 + utility)
        elif self.valuation_strategy == "conservative":
            return base_value * quality * utility
        else:  # balanced
            return base_value * (quality + utility)
    
    def win_item(self, item: Dict[str, Any], price: float):
        """Record won item"""
        self.won_items.append({
            "item": item,
            "price": price
        })
        self.remaining_budget -= price


class AuctionHouse:
    """Manages auction process"""
    
    def __init__(self, auction_type: str = "english"):
        self.auction_type = auction_type
        self.bidders = []
    
    def register_bidder(self, bidder: BiddingAgent):
        """Register bidder"""
        self.bidders.append(bidder)
    
    def conduct_auction(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Conduct auction for all items"""
        results = []
        
        for item in items:
            if self.auction_type == "english":
                result = self._english_auction(item)
            elif self.auction_type == "sealed_bid":
                result = self._sealed_bid_auction(item)
            else:  # dutch
                result = self._dutch_auction(item)
            
            results.append(result)
        
        return results
    
    def _english_auction(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Ascending price auction"""
        current_price = item.get("starting_price", 10)
        min_increment = item.get("min_increment", 5)
        max_rounds = 10
        
        bid_history = []
        
        for round_num in range(1, max_rounds + 1):
            # Get bids from all bidders
            bids = []
            for bidder in self.bidders:
                bid_amount = bidder.calculate_bid(item, current_price, round_num)
                if bid_amount > current_price:
                    bids.append({
                        "bidder_id": bidder.agent_id,
                        "amount": bid_amount
                    })
            
            if not bids:
                # No more bids, auction ends
                break
            
            # Find highest bid
            highest_bid = max(bids, key=lambda x: x["amount"])
            current_price = highest_bid["amount"]
            
            bid_history.append({
                "round": round_num,
                "bids": bids,
                "highest_bid": highest_bid,
                "current_price": current_price
            })
            
            # Check if only one bidder remains
            if len(bids) == 1:
                break
        
        # Determine winner
        if bid_history:
            final_round = bid_history[-1]
            winner_id = final_round["highest_bid"]["bidder_id"]
            final_price = final_round["highest_bid"]["amount"]
            
            # Award item to winner
            winner = next(b for b in self.bidders if b.agent_id == winner_id)
            winner.win_item(item, final_price)
            
            return {
                "item_id": item.get("id"),
                "item_name": item.get("name"),
                "winner": winner_id,
                "final_price": final_price,
                "rounds": len(bid_history),
                "total_bids": sum(len(r["bids"]) for r in bid_history),
                "bid_history": bid_history
            }
        
        return {
            "item_id": item.get("id"),
            "item_name": item.get("name"),
            "winner": None,
            "final_price": 0,
            "rounds": 0,
            "total_bids": 0,
            "bid_history": []
        }
    
    def _sealed_bid_auction(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Single round sealed bid auction"""
        bids = []
        
        for bidder in self.bidders:
            bid_amount = bidder.calculate_bid(item, 0, 1)
            if bid_amount > 0:
                bids.append({
                    "bidder_id": bidder.agent_id,
                    "amount": bid_amount
                })
        
        if bids:
            # Highest bidder wins
            highest_bid = max(bids, key=lambda x: x["amount"])
            winner_id = highest_bid["bidder_id"]
            final_price = highest_bid["amount"]
            
            winner = next(b for b in self.bidders if b.agent_id == winner_id)
            winner.win_item(item, final_price)
            
            return {
                "item_id": item.get("id"),
                "item_name": item.get("name"),
                "winner": winner_id,
                "final_price": final_price,
                "rounds": 1,
                "total_bids": len(bids),
                "all_bids": bids
            }
        
        return {
            "item_id": item.get("id"),
            "item_name": item.get("name"),
            "winner": None,
            "final_price": 0,
            "rounds": 1,
            "total_bids": 0,
            "all_bids": []
        }
    
    def _dutch_auction(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Descending price auction"""
        starting_price = item.get("base_value", 100) * 1.5
        min_price = item.get("starting_price", 10)
        price_decrement = 10
        
        current_price = starting_price
        round_num = 0
        
        while current_price >= min_price:
            round_num += 1
            
            # Check if any bidder accepts current price
            for bidder in self.bidders:
                estimated_value = bidder._estimate_value(item)
                if current_price <= estimated_value and current_price <= bidder.remaining_budget:
                    # First bidder to accept wins
                    bidder.win_item(item, current_price)
                    return {
                        "item_id": item.get("id"),
                        "item_name": item.get("name"),
                        "winner": bidder.agent_id,
                        "final_price": current_price,
                        "rounds": round_num,
                        "starting_price": starting_price
                    }
            
            current_price -= price_decrement
        
        # No winner
        return {
            "item_id": item.get("id"),
            "item_name": item.get("name"),
            "winner": None,
            "final_price": 0,
            "rounds": round_num,
            "starting_price": starting_price
        }


def setup_auction_agent(state: AgentAuctionState) -> AgentAuctionState:
    """Setup auction"""
    print("\n🏛️ Setting Up Auction...")
    
    items = [
        {
            "id": "ITEM_1",
            "name": "Premium Cloud Credits",
            "base_value": 100,
            "quality": 0.9,
            "utility": 0.8,
            "starting_price": 20,
            "min_increment": 5
        },
        {
            "id": "ITEM_2",
            "name": "Data Processing Package",
            "base_value": 80,
            "quality": 0.7,
            "utility": 0.9,
            "starting_price": 15,
            "min_increment": 5
        },
        {
            "id": "ITEM_3",
            "name": "Support Contract",
            "base_value": 60,
            "quality": 0.8,
            "utility": 0.6,
            "starting_price": 10,
            "min_increment": 5
        }
    ]
    
    bidders_config = [
        {"id": "Bidder_Alpha", "budget": 250, "strategy": "aggressive"},
        {"id": "Bidder_Beta", "budget": 200, "strategy": "balanced"},
        {"id": "Bidder_Gamma", "budget": 180, "strategy": "conservative"},
        {"id": "Bidder_Delta", "budget": 220, "strategy": "balanced"}
    ]
    
    print(f"\n  Auction Items: {len(items)}")
    for item in items:
        print(f"\n    {item['name']} ({item['id']}):")
        print(f"      Base Value: ${item['base_value']}")
        print(f"      Starting Price: ${item['starting_price']}")
        print(f"      Quality: {item['quality']:.0%}")
        print(f"      Utility: {item['utility']:.0%}")
    
    print(f"\n  Registered Bidders: {len(bidders_config)}")
    for bidder in bidders_config:
        print(f"    • {bidder['id']}: Budget=${bidder['budget']}, Strategy={bidder['strategy']}")
    
    return {
        **state,
        "auction_items": items,
        "bidders": bidders_config,
        "messages": [f"✓ Auction setup: {len(items)} items, {len(bidders_config)} bidders"]
    }


def run_auction_agent(state: AgentAuctionState) -> AgentAuctionState:
    """Run auction"""
    print("\n🔨 Running Auction...")
    
    # Create auction house
    auction = AuctionHouse(auction_type="english")
    
    # Register bidders
    for bidder_config in state["bidders"]:
        bidder = BiddingAgent(
            bidder_config["id"],
            bidder_config["budget"],
            bidder_config["strategy"]
        )
        auction.register_bidder(bidder)
    
    # Conduct auction
    results = auction.conduct_auction(state["auction_items"])
    
    print(f"\n  Auction Type: English (Ascending Price)")
    print(f"  Total Items: {len(results)}")
    
    for result in results:
        print(f"\n  {result['item_name']}:")
        if result["winner"]:
            print(f"    Winner: {result['winner']}")
            print(f"    Final Price: ${result['final_price']}")
            print(f"    Rounds: {result['rounds']}")
            print(f"    Total Bids: {result['total_bids']}")
        else:
            print(f"    Status: No winner")
    
    return {
        **state,
        "auction_results": results,
        "messages": [f"✓ Auction completed: {len(results)} items sold"]
    }


def generate_auction_report_agent(state: AgentAuctionState) -> AgentAuctionState:
    """Generate auction report"""
    print("\n" + "="*70)
    print("AGENT AUCTION REPORT")
    print("="*70)
    
    print(f"\n🏛️ Auction Details:")
    print(f"  Total Items: {len(state['auction_items'])}")
    print(f"  Total Bidders: {len(state['bidders'])}")
    print(f"  Auction Type: English (Ascending Price)")
    
    print(f"\n📦 Auction Items:")
    for item in state["auction_items"]:
        print(f"\n  {item['name']}:")
        print(f"    Base Value: ${item['base_value']}")
        print(f"    Starting Price: ${item['starting_price']}")
        print(f"    Quality: {item['quality']:.0%}, Utility: {item['utility']:.0%}")
    
    print(f"\n👥 Bidders:")
    for bidder in state["bidders"]:
        print(f"  • {bidder['id']}: Budget=${bidder['budget']}, Strategy={bidder['strategy']}")
    
    print(f"\n🔨 Auction Results:")
    items_sold = sum(1 for r in state["auction_results"] if r.get("winner"))
    total_revenue = sum(r.get("final_price", 0) for r in state["auction_results"])
    
    for result in state["auction_results"]:
        print(f"\n  {result['item_name']}:")
        
        if result.get("winner"):
            print(f"    ✅ SOLD")
            print(f"    Winner: {result['winner']}")
            print(f"    Final Price: ${result['final_price']}")
            print(f"    Auction Rounds: {result['rounds']}")
            print(f"    Total Bids Received: {result['total_bids']}")
            
            if "bid_history" in result and result["bid_history"]:
                print(f"\n    Bidding History:")
                for round_data in result["bid_history"][-3:]:  # Show last 3 rounds
                    print(f"      Round {round_data['round']}: {len(round_data['bids'])} bid(s), "
                          f"High=${round_data['highest_bid']['amount']}")
        else:
            print(f"    ❌ UNSOLD")
            print(f"    Reason: No qualifying bids")
    
    print(f"\n📊 Auction Statistics:")
    print(f"  Items Sold: {items_sold}/{len(state['auction_items'])} ({items_sold/len(state['auction_items'])*100:.0f}%)")
    print(f"  Total Revenue: ${total_revenue}")
    print(f"  Average Sale Price: ${total_revenue/items_sold if items_sold > 0 else 0:.2f}")
    
    total_rounds = sum(r.get("rounds", 0) for r in state["auction_results"])
    total_bids = sum(r.get("total_bids", 0) for r in state["auction_results"])
    print(f"  Total Rounds: {total_rounds}")
    print(f"  Total Bids: {total_bids}")
    print(f"  Average Bids per Item: {total_bids/len(state['auction_items']):.1f}")
    
    print(f"\n💡 Agent Auction Benefits:")
    print("  • Competitive price discovery")
    print("  • Fair and transparent process")
    print("  • Automated allocation")
    print("  • Market-driven pricing")
    print("  • Efficient resource distribution")
    print("  • Multiple auction protocols")
    print("  • Strategy flexibility")
    
    print("\n="*70)
    print("✅ Agent Auction Pattern Complete!")
    print("="*70)
    
    return {**state, "messages": ["✓ Report generated"]}


def create_agent_auction_graph():
    workflow = StateGraph(AgentAuctionState)
    workflow.add_node("setup", setup_auction_agent)
    workflow.add_node("auction", run_auction_agent)
    workflow.add_node("report", generate_auction_report_agent)
    workflow.add_edge(START, "setup")
    workflow.add_edge("setup", "auction")
    workflow.add_edge("auction", "report")
    workflow.add_edge("report", END)
    return workflow.compile()


def main():
    print("="*70)
    print("Pattern 270: Agent Auction MCP Pattern")
    print("="*70)
    
    app = create_agent_auction_graph()
    final_state = app.invoke({
        "messages": [],
        "auction_items": [],
        "bidders": [],
        "auction_rounds": [],
        "auction_results": []
    })
    print("\n✅ Agent Auction Pattern Complete!")


if __name__ == "__main__":
    main()
