"""
Report download routes.

Two endpoints:
  GET /reports/<id>/pdf   — stream a PDF report for PredictionResult <id>
  GET /reports/<id>/excel — stream an Excel (.xlsx) report for PredictionResult <id>

Both:
  - Require authentication (@login_required)
  - Verify the result belongs to the current user (ownership check)
  - Return the file as an attachment (browser download dialog)
  - Generate the file on-the-fly from DB data; nothing is stored on disk
"""

from flask import Blueprint, Response, abort
from flask_login import login_required, current_user

from app.models_db import PredictionResult
from app.utils.report_generator import generate_pdf, generate_excel

reports_bp = Blueprint('reports', __name__)


@reports_bp.route('/reports/<int:result_id>/pdf')
@login_required
def download_pdf(result_id: int):
    """
    Generate and stream a PDF report for the given PredictionResult.

    Security: result is fetched with both id AND user_id filter so
    one user cannot download another user's report by guessing an id.
    """
    result = PredictionResult.query.filter_by(
        id=result_id,
        user_id=current_user.id
    ).first_or_404()

    pdf_bytes = generate_pdf(result)

    # 'attachment' triggers the browser save-file dialog
    # Content-Disposition filename is sanitised (no user-controlled strings in it)
    return Response(
        pdf_bytes,
        mimetype='application/pdf',
        headers={
            'Content-Disposition': f'attachment; filename="prediction_report_{result_id}.pdf"',
            'Content-Length': str(len(pdf_bytes)),
        }
    )


@reports_bp.route('/reports/<int:result_id>/excel')
@login_required
def download_excel(result_id: int):
    """
    Generate and stream an Excel (.xlsx) report for the given PredictionResult.

    Same ownership check as the PDF endpoint.
    """
    result = PredictionResult.query.filter_by(
        id=result_id,
        user_id=current_user.id
    ).first_or_404()

    excel_bytes = generate_excel(result)

    return Response(
        excel_bytes,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={
            'Content-Disposition': f'attachment; filename="prediction_report_{result_id}.xlsx"',
            'Content-Length': str(len(excel_bytes)),
        }
    )
