"""Generate a detailed PDF report of all backtest results."""

import json
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether,
)

OUT = Path("output")
REPORT_PATH = OUT / "polymarket_backtest_report.pdf"

# Load metrics
metrics = {}
for name in ["scheduled_momentum", "liquidity_reversion", "whale_follow", "cross_market_arb", "resolution_catalyst"]:
    p = OUT / f"{name}_metrics.json"
    if p.exists():
        with open(p) as f:
            metrics[name] = json.load(f)


def build_report():
    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontSize=22, spaceAfter=6))
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["Normal"], fontSize=12, textColor=colors.grey, spaceAfter=12))
    styles.add(ParagraphStyle(name="SectionHead", parent=styles["Heading1"], fontSize=16, spaceBefore=16, spaceAfter=8, textColor=colors.HexColor("#1a1a2e")))
    styles.add(ParagraphStyle(name="SubHead", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#16213e")))
    styles.add(ParagraphStyle(name="BodyJ", parent=styles["Normal"], fontSize=10, leading=14, alignment=TA_JUSTIFY, spaceAfter=6))
    styles.add(ParagraphStyle(name="BulletItem", parent=styles["Normal"], fontSize=10, leading=14, leftIndent=20, bulletIndent=10, spaceAfter=3))
    styles.add(ParagraphStyle(name="SmallNote", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceAfter=4))

    story = []

    # ── TITLE PAGE ──
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("Polymarket Strategy Backtester", styles["TitleCustom"]))
    story.append(Paragraph("Multi-Strategy Backtest Report", styles["Subtitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"Report generated: {datetime.now().strftime('%B %d, %Y')}", styles["Subtitle"]))
    story.append(Paragraph("Data source: poly_data (github.com/warproxxx/poly_data)", styles["Subtitle"]))
    story.append(Paragraph("Backtest period: July 1 – October 1, 2024", styles["Subtitle"]))
    story.append(Paragraph("Trade events processed: ~8M (sampled every 5th → 1.59M)", styles["Subtitle"]))
    story.append(Spacer(1, 0.5 * inch))

    # Summary table
    summary_data = [
        ["Strategy", "Total PnL", "Win Rate", "Trades", "Sharpe", "Max DD"],
    ]
    for name, m in metrics.items():
        pnl = m.get("realized_pnl", 0)
        wr = m.get("win_rate", 0)
        tc = m.get("total_trades", 0)
        sh = m.get("sharpe", 0)
        mdd = m.get("max_drawdown_pct", 0)
        summary_data.append([
            name.replace("_", " ").title(),
            f"${pnl:+,.2f}",
            f"{wr:.1%}" if tc > 0 else "—",
            str(tc),
            f"{sh:.2f}" if tc > 0 else "—",
            f"{mdd:.1%}" if tc > 0 else "—",
        ])

    t = Table(summary_data, colWidths=[1.8*inch, 1.3*inch, 0.8*inch, 0.7*inch, 0.7*inch, 0.7*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(PageBreak())

    # ── EXECUTIVE SUMMARY ──
    story.append(Paragraph("1. Executive Summary", styles["SectionHead"]))
    story.append(Paragraph(
        "This report presents the results of backtesting five trading strategies against historical "
        "Polymarket prediction market data from July through September 2024. The backtester processes "
        "trade-level data from the poly_data dataset, which contains every fill on the Polymarket CLOB "
        "(Central Limit Order Book). We tested scheduled momentum, liquidity reversion, whale following, "
        "cross-market arbitrage, and resolution catalyst (market-making) strategies.",
        styles["BodyJ"]
    ))
    story.append(Paragraph(
        "Of the five strategies, only three generated trades. <b>Liquidity Reversion</b> showed "
        "apparently exceptional returns (+$216K from $10K, 71% win rate), but deeper analysis reveals "
        "critical methodological flaws that inflate performance. <b>Whale Follow</b> generated 5 trades "
        "with 80% win rate — a promising signal with insufficient sample size. "
        "<b>Resolution Catalyst</b> lost money on 6 trades, suggesting adverse selection in the "
        "market-making approach.",
        styles["BodyJ"]
    ))

    # Strategy comparison chart
    img_path = OUT / "report_strategy_comparison.png"
    if img_path.exists():
        story.append(Spacer(1, 0.2 * inch))
        story.append(Image(str(img_path), width=6.5*inch, height=4.5*inch))
        story.append(Paragraph("Figure 1: Strategy comparison across all metrics.", styles["SmallNote"]))
    story.append(PageBreak())

    # ── METHODOLOGY ──
    story.append(Paragraph("2. Methodology", styles["SectionHead"]))

    story.append(Paragraph("2.1 Data", styles["SubHead"]))
    story.append(Paragraph(
        "The backtest uses the <b>poly_data</b> dataset, an open-source archive of all Polymarket "
        "CLOB activity. The dataset contains:",
        styles["BodyJ"]
    ))
    for item in [
        "<b>markets.csv</b> (50 MB): 119,554 markets with metadata (question, close time, resolution)",
        "<b>processed/trades.csv</b> (33 GB): Every matched trade with timestamp, price, size, maker/taker addresses, and direction",
        "Date range filtered to <b>July 1 – October 1, 2024</b> yielding 7.95M raw trade events",
        "Sampled every 5th trade for performance → <b>1,590,155</b> trade events processed per strategy",
    ]:
        story.append(Paragraph(f"• {item}", styles["BulletItem"]))

    story.append(Paragraph("2.2 Backtest Engine", styles["SubHead"]))
    story.append(Paragraph(
        "The engine is event-driven: it iterates through trades chronologically, updates per-market "
        "state (VWAP, volume, spread estimate via incremental accumulators), and calls strategy hooks. "
        "Key parameters:",
        styles["BodyJ"]
    ))
    params_data = [
        ["Parameter", "Value"],
        ["Initial Cash", "$10,000"],
        ["Slippage", "$0.01 per share"],
        ["Max Position Size", "$500 per trade"],
        ["Max Open Positions", "20 concurrent"],
        ["Equity Snapshots", "Every 1 hour"],
        ["Resolution", "Open positions resolved at market close"],
    ]
    pt = Table(params_data, colWidths=[2.5*inch, 2*inch])
    pt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(pt)

    story.append(Paragraph("2.3 Strategies", styles["SubHead"]))
    strat_descriptions = [
        ("<b>Scheduled Momentum</b>: Buy near-certain markets (YES price $0.70–$0.97) within 24h of resolution when trading at a discount to model fair value. Hold to resolution."),
        ("<b>Liquidity Reversion</b>: Fade large price impacts (>5%) in low-volume markets (<$50K 24h volume). Buy the opposite side expecting mean reversion within 4 hours."),
        ("<b>Whale Follow</b>: Detect large trades (>$5K USD) and follow with a 15-minute delay. Take profit/stop loss at ±5%."),
        ("<b>Cross-Market Arb</b>: Exploit monotonicity inversions in related markets within the same event (e.g., \"BTC >100K\" should be priced higher than \"BTC >110K\")."),
        ("<b>Resolution Catalyst</b>: Provide liquidity (market-make) in the 48–2 hour window before resolution, earning the bid-ask spread."),
    ]
    for desc in strat_descriptions:
        story.append(Paragraph(f"• {desc}", styles["BulletItem"]))
    story.append(PageBreak())

    # ── STRATEGY RESULTS ──
    story.append(Paragraph("3. Strategy Results", styles["SectionHead"]))

    # ── 3.1 Liquidity Reversion (the main one) ──
    story.append(Paragraph("3.1 Liquidity Reversion — Detailed Analysis", styles["SubHead"]))

    m = metrics.get("liquidity_reversion", {})
    story.append(Paragraph(
        f"The liquidity reversion strategy generated <b>{m.get('total_trades', 0):,} closed trades</b> "
        f"with a reported PnL of <b>${m.get('realized_pnl', 0):+,.2f}</b> and a win rate of "
        f"<b>{m.get('win_rate', 0):.1%}</b>. However, these headline numbers are misleading due to "
        "several methodological issues identified in post-hoc analysis.",
        styles["BodyJ"]
    ))

    # Metrics table
    lr_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Total Trades", f"{m.get('total_trades', 0):,}", "Sharpe Ratio", f"{m.get('sharpe', 0):.2f}"],
        ["Win Rate", f"{m.get('win_rate', 0):.1%}", "Max Drawdown", f"{m.get('max_drawdown_pct', 0):.1%}"],
        ["Avg Winner", f"${m.get('avg_winner', 0):+.2f}", "Avg Loser", f"${m.get('avg_loser', 0):+.2f}"],
        ["Best Trade", f"${m.get('best_trade', 0):+.2f}", "Worst Trade", f"${m.get('worst_trade', 0):+.2f}"],
        ["Mean Return", f"{m.get('mean_return', 0):.1%}", "Median Return", f"{m.get('median_return', 0):.1%}"],
        ["Avg Hold", f"{m.get('avg_hold_hours', 0):.1f}h", "Turnover", f"{m.get('turnover', 0):.1f}x"],
    ]
    lt = Table(lr_data, colWidths=[1.4*inch, 1.2*inch, 1.4*inch, 1.2*inch])
    lt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f0f0")]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(lt)
    story.append(Spacer(1, 0.1 * inch))

    # Equity curve
    eq_path = OUT / "liquidity_reversion_equity.png"
    if eq_path.exists():
        story.append(Image(str(eq_path), width=6*inch, height=2.5*inch))
        story.append(Paragraph("Figure 2: Liquidity Reversion equity curve.", styles["SmallNote"]))

    # Trade distribution
    td_path = OUT / "liquidity_reversion_trades.png"
    if td_path.exists():
        story.append(Image(str(td_path), width=6*inch, height=2.5*inch))
        story.append(Paragraph("Figure 3: PnL distribution and cumulative PnL.", styles["SmallNote"]))
    story.append(PageBreak())

    # ── Critical Analysis ──
    story.append(Paragraph("3.1.1 Why These Results Are Unrealistic", styles["SubHead"]))
    story.append(Paragraph(
        "Post-hoc analysis reveals three compounding problems that inflate the strategy's apparent "
        "performance far beyond what would be achievable in live trading:",
        styles["BodyJ"]
    ))

    story.append(Paragraph("<b>Problem 1: Cheap Token Leverage (Primary Issue)</b>", styles["BodyJ"]))
    story.append(Paragraph(
        "461 of 1,121 trades (41%) enter at prices below $0.15. When the strategy buys YES at $0.08 "
        "with a $300 position budget, it acquires 3,750 shares. If the market subsequently trades at "
        "$0.50 — a normal level of volatility for thin prediction markets, not necessarily a "
        "\"reversion\" — the position is worth $1,875, yielding a $1,575 profit (525% return) from "
        "a $300 investment. These are not temporary dislocations reverting to fair value; they are "
        "cheap, volatile markets where any upward move generates outsized returns due to the "
        "shares-based position sizing. <b>$135K of $216K total profit (63%) comes from sub-$0.15 "
        "entries alone.</b>",
        styles["BodyJ"]
    ))

    # Entry price analysis chart
    ep_path = OUT / "report_entry_price_analysis.png"
    if ep_path.exists():
        story.append(Image(str(ep_path), width=6*inch, height=2.3*inch))
        story.append(Paragraph("Figure 4: Entry price distribution and PnL contribution by entry price bucket.", styles["SmallNote"]))

    story.append(Paragraph("<b>Problem 2: No Execution Realism</b>", styles["BodyJ"]))
    story.append(Paragraph(
        "The strategy observes a trade at price $X in the historical feed and immediately \"buys\" "
        "at that price. In reality, that trade has already been executed — the strategy would need "
        "to submit an order at a potentially worse price, compete for fills, and absorb market impact. "
        "The 5% impact threshold means every entry occurs precisely at the moment of maximum dislocation, "
        "which is an unrealistic assumption. In live trading, the strategy would either: (a) miss the "
        "fill entirely as the price reverts before the order executes, or (b) fill at a worse price "
        "that already reflects partial reversion, drastically reducing edge.",
        styles["BodyJ"]
    ))

    story.append(Paragraph("<b>Problem 3: Survivorship and Resolution Bias</b>", styles["BodyJ"]))
    story.append(Paragraph(
        "189 trades (17%) exit at prices above $0.90 or below $0.10, consistent with "
        "market resolution rather than mean reversion. These trades contribute $52K (24% of profits). "
        "The strategy accidentally holds positions through resolution events, capturing the full "
        "payoff spread rather than the intended reversion premium. Additionally, the top single "
        "market (ID 503313) generated $42K across 144 trades — 19% of total profit from one "
        "extremely volatile market with a price range of $0.13–$0.87.",
        styles["BodyJ"]
    ))

    # Market concentration chart
    mc_path = OUT / "report_market_concentration.png"
    if mc_path.exists():
        story.append(Image(str(mc_path), width=6*inch, height=2.3*inch))
        story.append(Paragraph("Figure 5: Market concentration — top 15 markets by trade count and PnL.", styles["SmallNote"]))
    story.append(PageBreak())

    # Return distribution
    rd_path = OUT / "report_return_distribution.png"
    if rd_path.exists():
        story.append(Image(str(rd_path), width=6*inch, height=2.3*inch))
        story.append(Paragraph("Figure 6: Return distribution and cheap token PnL contribution.", styles["SmallNote"]))

    # Side breakdown
    sb_path = OUT / "report_side_breakdown.png"
    if sb_path.exists():
        story.append(Image(str(sb_path), width=6*inch, height=2.3*inch))
        story.append(Paragraph("Figure 7: PnL and entry prices broken down by YES/NO side.", styles["SmallNote"]))

    # Hold time scatter
    ht_path = OUT / "report_holdtime_pnl.png"
    if ht_path.exists():
        story.append(Image(str(ht_path), width=5.5*inch, height=2.8*inch))
        story.append(Paragraph("Figure 8: Hold time vs PnL scatter — largest winners are short-hold cheap token trades.", styles["SmallNote"]))

    story.append(Paragraph("3.1.2 What Would Be Needed for Realistic Results", styles["SubHead"]))
    for fix in [
        "<b>Notional position sizing</b>: Cap the dollar value of shares held, not just the cost. A $300 position in a $0.05 token should be limited to $300 notional, not 6,000 shares worth $300 at entry but potentially $6,000 at exit.",
        "<b>Execution delay + slippage model</b>: Add 1–5 second execution delay and model the actual book depth. If a $50K trade moves the price 5%, a $300 follow-on order won't fill at the pre-move price.",
        "<b>Filter extreme prices</b>: Exclude markets with YES price below $0.20 or above $0.80 to avoid the cheap-token leverage effect and resolution proximity.",
        "<b>Full (unsampled) data</b>: The 5x sampling artificially inflates apparent price impacts when consecutive observed trades for a market span multiple real trades.",
        "<b>Realistic max drawdown constraints</b>: The 37% max drawdown on a $10K account represents $3,700 — an unacceptable loss for most traders.",
    ]:
        story.append(Paragraph(f"• {fix}", styles["BulletItem"]))
    story.append(PageBreak())

    # ── 3.2 Whale Follow ──
    story.append(Paragraph("3.2 Whale Follow", styles["SubHead"]))
    wm = metrics.get("whale_follow", {})
    story.append(Paragraph(
        f"The whale follow strategy generated <b>{wm.get('total_trades', 0)} trades</b> with a "
        f"win rate of <b>{wm.get('win_rate', 0):.0%}</b> and PnL of <b>${wm.get('realized_pnl', 0):+,.2f}</b>. "
        f"Without a curated whale address list, the strategy detected whales purely by trade size "
        f"(>{'>'}$5,000 USD). Only 5 signals met all criteria (size threshold, follow delay, price "
        f"stability, volume filter).",
        styles["BodyJ"]
    ))
    story.append(Paragraph(
        "The extremely small sample size (n=5) makes these results statistically meaningless. "
        "However, the directional signal — that large trades in low-volume markets predict "
        "short-term continuation — is consistent with market microstructure theory. "
        "To make this strategy viable, one would need: (a) a curated list of historically "
        "profitable wallets from on-chain analysis, (b) lower size thresholds to increase signal "
        "frequency, and (c) a longer backtest period for statistical significance.",
        styles["BodyJ"]
    ))

    wf_eq = OUT / "whale_follow_equity.png"
    if wf_eq.exists():
        story.append(Image(str(wf_eq), width=5*inch, height=2.1*inch))

    # ── 3.3 Resolution Catalyst ──
    story.append(Paragraph("3.3 Resolution Catalyst (Market-Making)", styles["SubHead"]))
    rm = metrics.get("resolution_catalyst", {})
    story.append(Paragraph(
        f"The resolution catalyst strategy made <b>{rm.get('total_trades', 0)} trades</b>, "
        f"losing <b>${abs(rm.get('realized_pnl', 0)):,.2f}</b> with a {rm.get('win_rate', 0):.0%} win rate. "
        f"The strategy attempts to provide liquidity in the 48–2 hour window before market resolution, "
        f"earning the bid-ask spread.",
        styles["BodyJ"]
    ))
    story.append(Paragraph(
        "The losses are consistent with adverse selection: informed traders are more active near "
        "resolution, and the market-maker gets picked off on the wrong side. The average loser "
        f"(${rm.get('avg_loser', 0):+.2f}) is much larger than the average winner "
        f"(${rm.get('avg_winner', 0):+.2f}), a classic market-making risk profile. This strategy "
        "would need a much tighter spread, better inventory management, and skew based on order "
        "flow toxicity to be viable.",
        styles["BodyJ"]
    ))

    rc_eq = OUT / "resolution_catalyst_equity.png"
    if rc_eq.exists():
        story.append(Image(str(rc_eq), width=5*inch, height=2.1*inch))

    # ── 3.4 Non-Trading Strategies ──
    story.append(Paragraph("3.4 Strategies That Did Not Trade", styles["SubHead"]))

    story.append(Paragraph("<b>Scheduled Momentum (0 trades)</b>", styles["BodyJ"]))
    story.append(Paragraph(
        "This strategy requires markets within 24 hours of resolution, priced between $0.70–$0.97, "
        "with at least $1,000 in 24h volume, where the price is below model fair value by at least 3%. "
        "While 4,331 markets closed during the test period, the edge model is extremely conservative: "
        "at 24 hours to resolution, fair value = $0.755, so only prices below $0.725 qualify. At "
        "shorter time horizons, fair value approaches $0.95, but the minimum price filter of $0.70 "
        "cuts off most candidates. The intersection of all filters produced zero qualifying trades.",
        styles["BodyJ"]
    ))

    story.append(Paragraph("<b>Cross-Market Arbitrage (0 trades)</b>", styles["BodyJ"]))
    story.append(Paragraph(
        "This strategy requires groups of 3+ related markets within the same event (e.g., \"BTC >100K\", "
        "\"BTC >110K\", \"BTC >120K\") to detect monotonicity inversions. The poly_data dataset does not "
        "include an event grouping field, and the strategy's on_init method — which should build event "
        "groups from market metadata — was not implemented against this data source. This strategy "
        "requires either an event taxonomy from the Polymarket API or NLP-based question clustering, "
        "neither of which was available in this backtest.",
        styles["BodyJ"]
    ))
    story.append(PageBreak())

    # ── CONCLUSIONS ──
    story.append(Paragraph("4. Conclusions & Recommendations", styles["SectionHead"]))

    story.append(Paragraph(
        "This backtest demonstrates both the promise and pitfalls of systematic trading on prediction "
        "markets. Key takeaways:",
        styles["BodyJ"]
    ))

    conclusions = [
        "<b>Liquidity reversion has a real signal, but execution matters enormously.</b> "
        "Thin prediction markets do exhibit mean reversion after large trades, but the magnitude "
        "of the edge is far smaller than the raw backtest suggests. Realistic execution modeling "
        "would likely reduce PnL by 80–90%.",
        "<b>Cheap token dynamics dominate returns.</b> "
        "Binary markets where tokens trade near $0 or $1 create asymmetric payoff profiles. "
        "Any strategy that systematically buys cheap tokens will appear profitable in backtests, "
        "but this is largely a mechanical artifact of leverage, not genuine alpha.",
        "<b>Whale following shows promise but needs on-chain data.</b> "
        "The size-based detection is too noisy. A curated wallet list from historical PnL analysis "
        "would dramatically improve signal quality.",
        "<b>Market-making near resolution is hazardous.</b> "
        "Adverse selection from informed traders overwhelms the spread premium. This strategy "
        "needs real-time order flow toxicity metrics to avoid getting picked off.",
        "<b>Cross-market arbitrage requires external data.</b> "
        "The strategy is theoretically sound but requires event grouping metadata not present "
        "in the trade-level data alone.",
    ]
    for c in conclusions:
        story.append(Paragraph(f"• {c}", styles["BulletItem"]))
        story.append(Spacer(1, 0.05 * inch))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("4.1 Recommended Next Steps", styles["SubHead"]))
    next_steps = [
        "Implement a realistic execution simulator with order book depth, latency, and partial fills",
        "Re-run liquidity reversion with notional position caps and price floor/ceiling filters",
        "Build a whale wallet scorer using historical on-chain PnL from Polymarket subgraph data",
        "Add event grouping via Polymarket API event_slug field or NLP clustering on market questions",
        "Extend backtest to 12+ months for statistical significance across market regimes",
        "Add transaction cost model: Polymarket charges 0% maker / 1% taker fees",
    ]
    for i, step in enumerate(next_steps, 1):
        story.append(Paragraph(f"{i}. {step}", styles["BulletItem"]))

    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("— End of Report —", styles["Subtitle"]))

    doc.build(story)
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    build_report()
