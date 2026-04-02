from abc import ABC, abstractmethod


def fmt_price(trieu: float) -> str:
    """Hiển thị giá: nếu >= 1000 triệu thì dùng tỷ, ngược lại giữ nguyên triệu."""
    ty = trieu / 1000
    if ty >= 1:
        return f"{ty:,.3f} tỷ"
    return f"{trieu:,.0f} triệu"


def fmt_price_m2(trieu_per_m2: float) -> str:
    """Hiển thị giá/m²: nếu >= 1000 triệu/m² thì dùng tỷ/m², ngược lại giữ triệu/m²."""
    ty = trieu_per_m2 / 1000
    if ty >= 1:
        return f"{ty:,.3f} tỷ/m²"
    return f"{trieu_per_m2:,.0f} triệu/m²"


class BasePage(ABC):
    """Base class cho tất cả các trang."""

    @abstractmethod
    def render(self) -> None:
        """Render nội dung trang."""
        ...
