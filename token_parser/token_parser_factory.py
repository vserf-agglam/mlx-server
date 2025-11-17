from token_parser.base_token_parser import BaseTokenParser, logger
from token_parser.qwen3_mode_parser import Qwen3MoeParser


class TokenParserFactory:
    """Factory for creating token parsers"""

    _parsers: dict[str, type[BaseTokenParser]] = {
        "qwen3_moe": Qwen3MoeParser,
    }

    @classmethod
    def create(cls, parser_name: str) -> BaseTokenParser:
        """
        Create a token parser instance.

        Args:
            parser_name: Name of the parser to create

        Returns:
            Token parser instance

        Raises:
            ValueError: If parser_name is not supported
        """
        parser_class = cls._parsers.get(parser_name)
        if parser_class is None:
            supported = ", ".join(cls._parsers.keys())
            raise ValueError(
                f"Unsupported token parser: {parser_name}. "
                f"Supported parsers: {supported}"
            )

        return parser_class()

    @classmethod
    def register_parser(cls, name: str, parser_class: type[BaseTokenParser]):
        """
        Register a custom token parser.

        Args:
            name: Name to register the parser under
            parser_class: Parser class to register
        """
        cls._parsers[name] = parser_class
        logger.info(f"Registered custom token parser: {name}")

    @classmethod
    def list_parsers(cls) -> list[str]:
        """Get list of available parser names"""
        return list(cls._parsers.keys())