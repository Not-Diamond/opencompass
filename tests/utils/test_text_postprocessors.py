from opencompass.opencompass.utils import text_postprocessors as undertest


def test_last_number_postprocess():
    s = "this string 123 contains some 13123 numbers 12312312 find the -1.75 last example 1337.0"
    assert undertest.last_number_postprocess(s) == 1337.0

    s2 = "Ili kujua Lloyd hupata pesa ngapi kwa wiki kutoka kwa mayai, tunahitaji kufanya mahesabu yafuatayo:\n\n1. Kwanza, tunahitaji kujua idadi ya madazeni ya mayai ambayo kuku wake hutaga kwa siku. Kwa kuwa dazeni moja lina mayai 12, tunagawanya idadi ya mayai yaliyotagwa kwa siku kwa 12:\n\n252 mayai / 12 mayai kwa dazeni = 21 dazeni\n\n2. Kisha, tunazidisha idadi ya madazeni kwa bei ya kila dazeni ili kupata mapato ya siku moja:\n\n21 dazeni x $2 kwa dazeni = $42 kwa siku\n\n3. Mwisho, tunazidisha mapato ya siku moja kwa idadi ya siku katika wiki ili kupata mapato ya wiki:\n\n$42 kwa siku x 7 siku kwa wiki = $294 kwa wiki\n\nKwa hivyo, Lloyd hupata $294 kutoka kwa mayai kila wiki.,"
    assert undertest.last_number_postprocess(s2) == 294
