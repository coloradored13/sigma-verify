"""Tests for sigma_verify.server — entrypoint coverage."""

from unittest.mock import MagicMock, patch


class TestMain:
    @patch("hateoas_agent.mcp_server.serve")
    @patch("sigma_verify.server.build_machine")
    def test_main_builds_machine_and_serves(self, mock_build, mock_serve):
        mock_machine = MagicMock()
        mock_build.return_value = mock_machine

        from sigma_verify.server import main

        main()

        mock_build.assert_called_once()
        mock_serve.assert_called_once_with(mock_machine, name="sigma-verify")
