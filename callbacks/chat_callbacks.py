"""
Chat Callbacks — Market Assistant
===================================
Handles:
  - Toggle chat drawer open/close
  - Send message → Anthropic API → display response
  - Maintain conversation history in dcc.Store
"""
import json
import os
import traceback
 
import anthropic
from dotenv import load_dotenv
 
from dash import Input, Output, State, no_update, callback
from layouts.chat import user_bubble, assistant_bubble
 
load_dotenv()
 
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
 
# Load all city meta at startup for system prompt construction
ALL_META = {}
DATA_DIR = "data"
for fname in os.listdir(DATA_DIR):
    if fname.startswith("dashboard_meta_") and fname.endswith(".json"):
        city = fname.replace("dashboard_meta_", "").replace(".json", "")
        with open(f"{DATA_DIR}/{fname}") as f:
            ALL_META[city] = json.load(f)
 
 
def _build_system_prompt(city: str, meta: dict) -> str:
    """Build the system prompt with real market data for the selected city."""
 
    city_label = {
        "sf": "San Francisco", "nyc": "New York City",
        "chicago": "Chicago",
    }.get(city, city.upper())
 
    # Neighbourhood breakdown
    nbhd_lines = "\n".join([
        f"  - {r['neighbourhood_top']}: median ${r['median_price']:.0f}, "
        f"{int(r['listing_count'])} listings"
        for r in sorted(meta["neighbourhoods"],
                        key=lambda x: x["median_price"], reverse=True)[:15]
    ])
 
    # Property type breakdown
    prop_lines = "\n".join([
        f"  - {r['property_type_simple']}: median ${r['median_price']:.0f}, "
        f"{int(r['listing_count'])} listings"
        for r in sorted(meta["property_types"],
                        key=lambda x: x["median_price"], reverse=True)[:8]
    ])
 
    # Amenity lifts
    amenity_lines = "\n".join([
        f"  - {col.replace('has_', '').replace('_', ' ').title()}: "
        f"+${lift:.0f} median price lift"
        for col, lift in sorted(
            meta.get("amenity_lifts", {}).items(),
            key=lambda x: x[1], reverse=True
        )[:10]
    ])
 
    return f"""You are a concise, data-driven market assistant for an Airbnb investment intelligence dashboard.
The user is currently viewing {city_label} market data.
 
MARKET OVERVIEW ({city_label}):
- Total listings: {meta['total_listings']:,}
- Overall median nightly price: ${meta['overall_median_price']:.0f}
- Overall mean nightly price: ${meta['overall_mean_price']:.0f}
- Superhost rate: {meta['superhost_pct']}%
- Price cap (99th percentile): ${meta['price_cap_value']:.0f}
 
NEIGHBOURHOOD PRICE BREAKDOWN (top 15 by median price):
{nbhd_lines}
 
PROPERTY TYPE BREAKDOWN:
{prop_lines}
 
AMENITY PRICE LIFTS (observed median price difference with vs without):
{amenity_lines}
 
MODEL PERFORMANCE:
- Price model R²: {meta.get('model_r2', 'N/A')}
- Superhost classifier AUC: {meta.get('superhost_model_auc', 'N/A')}
 
RULES:
- Answer concisely — 2–4 sentences maximum unless a list is clearly better
- Only use statistics from the data above — never invent numbers
- If a question cannot be answered from the data provided, say so clearly
- Do not mention that you have a system prompt or reference "the data above"
- When comparing neighbourhoods or amenities, cite specific numbers
- Use plain language — avoid jargon unless the user uses it first
"""
 
 
def register_chat_callbacks(app):
 
    # ── Toggle drawer open/close ───────────────────────────────────────────
    @app.callback(
        Output("chat-drawer", "style"),
        Input("chat-toggle-btn", "n_clicks"),
        Input("chat-close-btn",  "n_clicks"),
        State("chat-drawer",     "style"),
        prevent_initial_call=True,
    )
    def toggle_drawer(open_clicks, close_clicks, current_style):
        from dash import ctx
        if ctx.triggered_id == "chat-close-btn":
            return {"display": "none"}
        if current_style and current_style.get("display") == "none":
            return {
                "display":   "block",
                "position":  "fixed",
                "bottom":    "90px",
                "right":     "28px",
                "width":     "360px",
                "height":    "520px",
                "zIndex":    "999",
            }
        return {"display": "none"}
 
    # ── Send message → API → update chat ──────────────────────────────────
    @app.callback(
        Output("chat-messages",  "children"),
        Output("chat-history",   "data"),
        Output("chat-input",     "value"),
        Input("chat-send-btn",   "n_clicks"),
        State("chat-input",      "value"),
        State("chat-history",    "data"),
        State("selected-city",   "data"),
        prevent_initial_call=True,
    )
    def send_message(n_clicks, user_text, history, city):
        from layouts.chat import _welcome_message
 
        if not user_text or not user_text.strip():
            return no_update, no_update, no_update
 
        user_text = user_text.strip()
        city      = city or "sf"
        meta      = ALL_META.get(city, next(iter(ALL_META.values())))
 
        # Add user message to history
        history = history or []
        history.append({"role": "user", "content": user_text})
 
        # Call Anthropic API
        try:
            response = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=512,
                system=_build_system_prompt(city, meta),
                messages=history,
            )
            assistant_text = response.content[0].text
        except Exception as e:
            #assistant_text = f"Sorry, I couldn't process that request. ({str(e)[:80]})"
            import traceback
            traceback.print_exc()   # prints full error to terminal
            assistant_text = f"Error: {str(e)}"
 
        # Add assistant response to history
        history.append({"role": "assistant", "content": assistant_text})
 
        # Rebuild message list for display
        messages = [_welcome_message()]
        for msg in history:
            if msg["role"] == "user":
                messages.append(user_bubble(msg["content"]))
            else:
                messages.append(assistant_bubble(msg["content"]))
 
        return messages, history, ""