"""
Chat drawer — floating AI assistant
"""
from dash import html, dcc
import dash_bootstrap_components as dbc
 
C = {
    "blue":  "#0071E3",
    "gray1": "#1D1D1F",
    "gray2": "#3A3A3C",
    "gray3": "#6E6E73",
    "gray4": "#AEAEB2",
    "gray5": "#C7C7CC",
    "gray6": "#F2F2F7",
}
 
 
def chat_components():
    """
    Returns the floating button + drawer as a single html.Div.
    Add this to app.layout — it renders on every page.
    """
    return html.Div([
 
        # ── Floating chat button ───────────────────────────────────────────
        html.Button(
            html.I(className="bi bi-chat-dots-fill",
                   style={"fontSize": "22px", "color": "#fff"}),
            id="chat-toggle-btn",
            n_clicks=0,
            style={
                "position":     "fixed",
                "bottom":       "28px",
                "right":        "28px",
                "width":        "52px",
                "height":       "52px",
                "borderRadius": "50%",
                "background":   C["blue"],
                "border":       "none",
                "boxShadow":    "0 4px 16px rgba(0,113,227,0.35)",
                "cursor":       "pointer",
                "zIndex":       "1000",
                "display":      "flex",
                "alignItems":   "center",
                "justifyContent": "center",
                "transition":   "transform 0.2s ease, box-shadow 0.2s ease",
            },
        ),
 
        # ── Chat drawer ────────────────────────────────────────────────────
        html.Div(
            id="chat-drawer",
            style={"display": "none"},
            children=html.Div([
 
                # Header
                html.Div([
                    html.Div([
                        html.Div(
                            html.I(className="bi bi-stars",
                                   style={"fontSize": "16px", "color": C["blue"]}),
                            style={
                                "width": "32px", "height": "32px",
                                "borderRadius": "50%",
                                "background": "#EBF5FB",
                                "display": "flex", "alignItems": "center",
                                "justifyContent": "center",
                            }
                        ),
                        html.Div([
                            html.P("Market Assistant",
                                   style={"fontSize": "14px", "fontWeight": "600",
                                          "color": C["gray1"], "margin": 0}),
                            html.P("Ask anything about the data",
                                   style={"fontSize": "11px", "color": C["gray4"],
                                          "margin": 0}),
                        ]),
                    ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
 
                    html.Button(
                        html.I(className="bi bi-x-lg",
                               style={"fontSize": "14px", "color": C["gray3"]}),
                        id="chat-close-btn",
                        n_clicks=0,
                        style={
                            "background": "none", "border": "none",
                            "cursor": "pointer", "padding": "4px",
                        },
                    ),
                ], style={
                    "display": "flex", "justifyContent": "space-between",
                    "alignItems": "center",
                    "padding": "16px 20px",
                    "borderBottom": f"1px solid {C['gray6']}",
                }),
 
                # Message history
                html.Div(
                    id="chat-messages",
                    children=[_welcome_message()],
                    style={
                        "flex": "1",
                        "overflowY": "auto",
                        "padding": "16px 20px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                    },
                ),
 
                # Loading indicator
                dcc.Loading(
                    html.Div(id="chat-loading-indicator"),
                    type="circle",
                    color=C["blue"],
                    style={"height": "20px"},
                ),
 
                # Input area
                html.Div([
                    dcc.Textarea(
                        id="chat-input",
                        placeholder="Ask about the market data...",
                        maxLength=500,
                        style={
                            "flex": "1",
                            "border": f"1px solid {C['gray5']}",
                            "borderRadius": "10px",
                            "padding": "10px 12px",
                            "fontSize": "13px",
                            "fontFamily": "-apple-system, BlinkMacSystemFont, sans-serif",
                            "resize": "none",
                            "outline": "none",
                            "height": "44px",
                            "lineHeight": "1.4",
                            "color": C["gray1"],
                        },
                    ),
                    html.Button(
                        html.I(className="bi bi-arrow-up",
                               style={"fontSize": "16px", "color": "#fff"}),
                        id="chat-send-btn",
                        n_clicks=0,
                        style={
                            "width": "36px", "height": "36px",
                            "borderRadius": "50%",
                            "background": C["blue"],
                            "border": "none",
                            "cursor": "pointer",
                            "display": "flex", "alignItems": "center",
                            "justifyContent": "center",
                            "flexShrink": "0",
                            "alignSelf": "flex-end",
                        },
                    ),
                ], style={
                    "display": "flex", "gap": "8px", "alignItems": "flex-end",
                    "padding": "12px 20px",
                    "borderTop": f"1px solid {C['gray6']}",
                }),
 
                # Conversation store
                dcc.Store(id="chat-history", data=[]),
 
            ], style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100%",
                "background": "#FFFFFF",
                "borderRadius": "16px",
                "boxShadow": "0 8px 40px rgba(0,0,0,0.12)",
                "overflow": "hidden",
            }),
            ),
 
    ])
 
 
def _welcome_message():
    return html.Div([
        html.Div(
            html.I(className="bi bi-stars",
                   style={"fontSize": "14px", "color": "#0071E3"}),
            style={
                "width": "28px", "height": "28px",
                "borderRadius": "50%", "background": "#EBF5FB",
                "display": "flex", "alignItems": "center",
                "justifyContent": "center", "flexShrink": "0",
            }
        ),
        html.Div(
            "Hi! I can answer questions about the market data — neighbourhoods, "
            "pricing, amenities, Superhost rates, and more. What would you like to know?",
            style={
                "background": "#F2F2F7",
                "borderRadius": "0 12px 12px 12px",
                "padding": "10px 14px",
                "fontSize": "13px",
                "color": "#3A3A3C",
                "lineHeight": "1.5",
                "maxWidth": "85%",
            }
        ),
    ], style={"display": "flex", "gap": "8px", "alignItems": "flex-start"})
 
 
def user_bubble(text: str):
    return html.Div(
        html.Div(
            text,
            style={
                "background": "#0071E3",
                "color": "#fff",
                "borderRadius": "12px 12px 0 12px",
                "padding": "10px 14px",
                "fontSize": "13px",
                "lineHeight": "1.5",
                "maxWidth": "85%",
            }
        ),
        style={"display": "flex", "justifyContent": "flex-end"}
    )
 
 
def assistant_bubble(text: str):
    return html.Div([
        html.Div(
            html.I(className="bi bi-stars",
                   style={"fontSize": "14px", "color": "#0071E3"}),
            style={
                "width": "28px", "height": "28px",
                "borderRadius": "50%", "background": "#EBF5FB",
                "display": "flex", "alignItems": "center",
                "justifyContent": "center", "flexShrink": "0",
            }
        ),
        html.Div(
            text,
            style={
                "background": "#F2F2F7",
                "borderRadius": "0 12px 12px 12px",
                "padding": "10px 14px",
                "fontSize": "13px",
                "color": "#3A3A3C",
                "lineHeight": "1.5",
                "maxWidth": "85%",
                "whiteSpace": "pre-wrap",
            }
        ),
    ], style={"display": "flex", "gap": "8px", "alignItems": "flex-start"})