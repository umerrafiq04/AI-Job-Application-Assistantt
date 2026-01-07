import gspread
from google.oauth2.service_account import Credentials


def get_gsheet_client():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_file(
        "backend/service_account.json",
        scopes=scopes
    )

    return gspread.authorize(creds)


def test_google_sheet_write():
    client = get_gsheet_client()

    # Open the sheet
    sheet = client.open("AI Job Applications").sheet1

    # Dummy test row
    test_row = [
        "TEST_COMPANY",
        "TEST_ROLE",
        "TEST_LOCATION",
        "99.99",
        "TEST_SOURCE",
        "https://example.com",
        "This is a test cover letter"
    ]

    sheet.append_row(test_row, value_input_option="RAW")

    print("✅ Test row inserted successfully!")


if __name__ == "__main__":
    test_google_sheet_write()
