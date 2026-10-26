"""Stand-in for flash-attention's rotary_emb extension: o1 = x1*cos - x2*sin, o2 = x1*sin + x2*cos."""


def apply_rotary(x1, x2, cos, sin, o1, o2, conj):
    if conj:
        sin = -sin
    a = x1 * cos - x2 * sin
    b = x1 * sin + x2 * cos
    o1.copy_(a)
    o2.copy_(b)
