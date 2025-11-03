# tests/test_reports.py
from app.utils import create_excel_report, create_pdf_report

def test_create_excel_report():
    snapshot = {"monthly_net_income":70000, "fixed_expenses":20000}
    sim_stats = {"prob_shortfall":0.2, "loan_payment":15000}
    b = create_excel_report(snapshot, sim_stats)
    assert isinstance(b, (bytes, bytearray))
    assert len(b) > 0

def test_create_pdf_report():
    snapshot = {"monthly_net_income":70000, "fixed_expenses":20000}
    sim_stats = {"prob_shortfall":0.2, "loan_payment":15000}
    b = create_pdf_report(snapshot, sim_stats, "BMW", 24)
    assert isinstance(b, (bytes, bytearray))
    assert len(b) > 0
