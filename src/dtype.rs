use crate::core::BitCollection;
use crate::enums::{ByteOrder, DtypeKind};
use crate::helpers::validate_slice;
use crate::iterator::ValuesIterator;
use crate::tibs_::{Tibs, bv_from_value, bv_from_values_iter, py_from_value, py_values_from_range};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyTuple, PyType};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct SingleDtype {
    pub(crate) kind: DtypeKind,
    pub(crate) length: usize,
    pub(crate) byte_order: ByteOrder,
}

impl SingleDtype {
    fn from_parts(kind: DtypeKind, length: i64, byte_order: ByteOrder) -> PyResult<Self> {
        if length <= 0 {
            return Err(PyValueError::new_err(format!(
                "Dtype length must be greater than zero, but received {length}."
            )));
        }
        let length = length as usize;
        if kind == DtypeKind::Bool && length != 1 {
            return Err(PyValueError::new_err(format!(
                "A Dtype of kind {} must have length 1.",
                kind.repr_name()
            )));
        }
        match kind {
            DtypeKind::Float if !matches!(length, 16 | 32 | 64) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have length 16, 32 or 64 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            DtypeKind::Bytes if !length.is_multiple_of(8) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have a length that is a multiple of 8 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            DtypeKind::Hex if !length.is_multiple_of(4) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have a length that is a multiple of 4 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            DtypeKind::Oct if !length.is_multiple_of(3) => {
                return Err(PyValueError::new_err(format!(
                    "A Dtype of kind {} must have a length that is a multiple of 3 bits. Received {length}.",
                    kind.repr_name()
                )));
            }
            _ => {}
        }
        if byte_order != ByteOrder::Unspecified {
            match kind {
                DtypeKind::Uint | DtypeKind::Int | DtypeKind::Float => {
                    if !length.is_multiple_of(8) {
                        return Err(PyValueError::new_err(format!(
                            "If a Dtype byte_order is given, the length must be a multiple of 8 (length = {length})."
                        )));
                    }
                }
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "A byte order cannot be specified for a Dtype of kind {}.",
                        kind.repr_name()
                    )));
                }
            }
        }
        Ok(Self {
            kind,
            length,
            byte_order,
        })
    }

    fn parse(spec: &str) -> PyResult<Self> {
        let spec = spec.trim().to_ascii_lowercase();
        let (base, byte_order) = if let Some(base) = spec.strip_suffix("_le") {
            (base, ByteOrder::Little)
        } else if let Some(base) = spec.strip_suffix("_be") {
            (base, ByteOrder::Big)
        } else {
            (spec.as_str(), ByteOrder::Unspecified)
        };

        if base == "bool" {
            return Self::from_parts(DtypeKind::Bool, 1, byte_order);
        }

        let (kind, length_text) = if let Some(length) = base.strip_prefix("bytes") {
            (DtypeKind::Bytes, length)
        } else if let Some(length) = base.strip_prefix("bits") {
            (DtypeKind::Bits, length)
        } else if let Some(length) = base.strip_prefix("bin") {
            (DtypeKind::Bin, length)
        } else if let Some(length) = base.strip_prefix("oct") {
            (DtypeKind::Oct, length)
        } else if let Some(length) = base.strip_prefix("hex") {
            (DtypeKind::Hex, length)
        } else if let Some(length) = base.strip_prefix('u') {
            (DtypeKind::Uint, length)
        } else if let Some(length) = base.strip_prefix('i') {
            (DtypeKind::Int, length)
        } else if let Some(length) = base.strip_prefix('f') {
            (DtypeKind::Float, length)
        } else {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}'."
            )));
        };

        if length_text.is_empty() || !length_text.chars().all(|c| c.is_ascii_digit()) {
            return Err(PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': missing or invalid bit length."
            )));
        }
        let length = length_text.parse::<i64>().map_err(|_| {
            PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{spec}': bit length is too large."
            ))
        })?;
        Self::from_parts(kind, length, byte_order)
    }

    fn spec(&self) -> String {
        let byte_order = match self.byte_order {
            ByteOrder::Unspecified => "",
            ByteOrder::Little => "_le",
            ByteOrder::Big => "_be",
        };
        match self.kind {
            DtypeKind::Uint => format!("u{}{byte_order}", self.length),
            DtypeKind::Int => format!("i{}{byte_order}", self.length),
            DtypeKind::Float => format!("f{}{byte_order}", self.length),
            DtypeKind::Bool => "bool".to_string(),
            DtypeKind::Bits => format!("bits{}", self.length),
            DtypeKind::Bin => format!("bin{}", self.length),
            DtypeKind::Oct => format!("oct{}", self.length),
            DtypeKind::Hex => format!("hex{}", self.length),
            DtypeKind::Bytes => format!("bytes{}", self.length),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum DtypeRepr {
    Single(SingleDtype),
    Array { dtype: Box<DtypeRepr>, count: usize },
    Tuple(Vec<DtypeRepr>),
}

impl DtypeRepr {
    pub(crate) fn length(&self) -> PyResult<usize> {
        match self {
            Self::Single(dtype) => Ok(dtype.length),
            Self::Array { dtype, count } => dtype
                .length()?
                .checked_mul(*count)
                .ok_or_else(|| PyValueError::new_err("Dtype length is too large to represent.")),
            Self::Tuple(dtypes) => dtypes.iter().try_fold(0usize, |total, dtype| {
                total
                    .checked_add(dtype.length()?)
                    .ok_or_else(|| PyValueError::new_err("Dtype length is too large to represent."))
            }),
        }
    }

    fn spec(&self) -> String {
        match self {
            Self::Single(dtype) => dtype.spec(),
            Self::Array { dtype, count } => format!("[{}; {count}]", dtype.spec()),
            Self::Tuple(dtypes) if dtypes.len() == 1 => {
                format!("({},)", dtypes[0].spec())
            }
            Self::Tuple(dtypes) => {
                let fields = dtypes.iter().map(Self::spec).collect::<Vec<_>>().join(", ");
                format!("({fields})")
            }
        }
    }
}

/// One field of a precomputed flat record layout: its kind/length/byte_order
/// plus its bit offset within one record. See [`RecordLayout`].
#[derive(Clone, Copy)]
pub(crate) struct RecordField {
    pub(crate) kind: DtypeKind,
    pub(crate) length: usize,
    pub(crate) byte_order: ByteOrder,
    pub(crate) bit_offset: usize,
}

/// A precomputed flat layout for a [`DtypeRepr::Tuple`] whose fields are all
/// [`DtypeRepr::Single`], or a [`DtypeRepr::Array`] whose element is
/// [`DtypeRepr::Single`] — the "record of scalar fields" shape (e.g.
/// `struct`'s `">hhl"`, or an MPEG-header-style fixed table). It lets pack and
/// unpack address each field directly instead of re-walking `DtypeRepr` and
/// recomputing offsets via `DtypeRepr::length()` on every single record.
///
/// Deeper nesting (tuple-of-tuple, array-of-tuple, ...) is never represented
/// here: `Dtype::record_layout` is `None` for those, and pack/unpack keep
/// walking `DtypeRepr` recursively exactly as before.
///
/// `Array` stores one element descriptor plus `count`, not `count` cloned
/// entries, so a large array stays cheap to represent: `Dtype("[u8; 1_000_000]")`
/// is as cheap to build as `Dtype("[u8; 4]")`. `Tuple` does flatten its fields
/// literally, which is safe because that count is bounded by the dtype spec's
/// own field arity, not by data volume.
pub(crate) enum RecordLayout {
    Tuple(Vec<RecordField>),
    Array { element: RecordField, count: usize },
}

fn build_record_layout(repr: &DtypeRepr) -> Option<RecordLayout> {
    match repr {
        DtypeRepr::Tuple(dtypes) => {
            let mut fields = Vec::with_capacity(dtypes.len());
            let mut bit_offset = 0;
            for dtype in dtypes {
                let DtypeRepr::Single(single) = dtype else {
                    return None;
                };
                fields.push(RecordField {
                    kind: single.kind,
                    length: single.length,
                    byte_order: single.byte_order,
                    bit_offset,
                });
                bit_offset += single.length;
            }
            Some(RecordLayout::Tuple(fields))
        }
        DtypeRepr::Array { dtype, count } => {
            let DtypeRepr::Single(single) = dtype.as_ref() else {
                return None;
            };
            Some(RecordLayout::Array {
                element: RecordField {
                    kind: single.kind,
                    length: single.length,
                    byte_order: single.byte_order,
                    bit_offset: 0,
                },
                count: *count,
            })
        }
        DtypeRepr::Single(_) => None,
    }
}

struct DtypeParser<'a> {
    spec: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> DtypeParser<'a> {
    fn new(spec: &'a str) -> Self {
        Self {
            spec,
            bytes: spec.as_bytes(),
            pos: 0,
        }
    }

    fn parse(mut self) -> PyResult<DtypeRepr> {
        let dtype = self.parse_dtype()?;
        self.skip_whitespace();
        if self.pos != self.bytes.len() {
            return self.error("unexpected trailing text");
        }
        Ok(dtype)
    }

    fn parse_dtype(&mut self) -> PyResult<DtypeRepr> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'[') => self.parse_array(),
            Some(b'(') => self.parse_tuple(),
            Some(_) => self.parse_single(),
            None => self.error("expected a dtype"),
        }
    }

    fn parse_single(&mut self) -> PyResult<DtypeRepr> {
        let start = self.pos;
        while let Some(byte) = self.peek() {
            if byte.is_ascii_whitespace() || matches!(byte, b'[' | b']' | b'(' | b')' | b',' | b';')
            {
                break;
            }
            self.pos += 1;
        }
        if self.pos == start {
            return self.error("expected a scalar dtype");
        }
        Ok(DtypeRepr::Single(SingleDtype::parse(
            &self.spec[start..self.pos],
        )?))
    }

    fn parse_array(&mut self) -> PyResult<DtypeRepr> {
        self.pos += 1;
        let dtype = self.parse_dtype()?;
        self.skip_whitespace();
        self.expect(b';', "expected ';' between the array dtype and count")?;
        self.skip_whitespace();
        let start = self.pos;
        while matches!(self.peek(), Some(byte) if byte.is_ascii_digit()) {
            self.pos += 1;
        }
        if start == self.pos {
            return self.error("expected a positive array count");
        }
        let count = self.spec[start..self.pos].parse::<usize>().map_err(|_| {
            PyValueError::new_err(format!(
                "Cannot parse Dtype spec '{}': array count is too large.",
                self.spec
            ))
        })?;
        if count == 0 {
            return self.error("array count must be greater than zero");
        }
        self.skip_whitespace();
        self.expect(b']', "expected ']' after the array count")?;
        let repr = DtypeRepr::Array {
            dtype: Box::new(dtype),
            count,
        };
        repr.length()?;
        Ok(repr)
    }

    fn parse_tuple(&mut self) -> PyResult<DtypeRepr> {
        self.pos += 1;
        self.skip_whitespace();
        if self.peek() == Some(b')') {
            return self.error("tuple dtypes must contain at least one dtype");
        }

        let mut dtypes = vec![self.parse_dtype()?];
        self.skip_whitespace();
        self.expect(b',', "a one-element tuple dtype needs a trailing comma")?;

        loop {
            self.skip_whitespace();
            if self.peek() == Some(b')') {
                self.pos += 1;
                break;
            }
            dtypes.push(self.parse_dtype()?);
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b')') if dtypes.len() > 1 => {
                    self.pos += 1;
                    break;
                }
                _ => return self.error("expected ',' or ')' after tuple dtype"),
            }
        }

        let repr = DtypeRepr::Tuple(dtypes);
        repr.length()?;
        Ok(repr)
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(byte) if byte.is_ascii_whitespace()) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn expect(&mut self, expected: u8, message: &str) -> PyResult<()> {
        if self.peek() != Some(expected) {
            return self.error(message);
        }
        self.pos += 1;
        Ok(())
    }

    fn error<T>(&self, message: &str) -> PyResult<T> {
        Err(PyValueError::new_err(format!(
            "Cannot parse Dtype spec '{}': {message} at position {}.",
            self.spec, self.pos
        )))
    }
}

///     The base class for fixed-width value descriptors.
///
///     Constructing ``Dtype`` parses ``spec`` and returns a
///     :class:`DtypeSingle`, :class:`DtypeArray` or :class:`DtypeTuple`.
///
///     :param str spec: A scalar, array or tuple dtype specification.
///     :return: The corresponding concrete dtype.
///
///     .. code-block:: pycon
///
///         >>> Dtype("u16_le")
///         DtypeSingle('u16_le')
///         >>> Dtype("[(u8, bool); 2]")
///         DtypeArray('[(u8, bool); 2]')
///
#[pyclass(module = "tibs", frozen, subclass, skip_from_py_object)]
#[derive(Clone)]
pub struct Dtype {
    pub(crate) repr: DtypeRepr,
    pub(crate) length: usize,
    pub(crate) record_layout: Option<Arc<RecordLayout>>,
}

// `length` and `record_layout` are both pure functions of `repr`, so equality
// and hashing are defined over `repr` alone rather than derived over every
// field — deriving would need `RecordLayout` to carry its own `PartialEq`/
// `Hash` for no semantic benefit, since two dtypes with equal `repr` always
// have equal `length`/`record_layout` already.
impl PartialEq for Dtype {
    fn eq(&self, other: &Self) -> bool {
        self.repr == other.repr
    }
}

impl Eq for Dtype {}

impl Hash for Dtype {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.repr.hash(state);
    }
}

impl Dtype {
    fn from_repr(repr: DtypeRepr) -> PyResult<Self> {
        let length = repr.length()?;
        let record_layout = build_record_layout(&repr).map(Arc::new);
        Ok(Self {
            repr,
            length,
            record_layout,
        })
    }

    fn parse_spec(spec: &str) -> PyResult<Self> {
        Self::from_repr(DtypeParser::new(spec).parse()?)
    }

    fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.repr {
            DtypeRepr::Single(_) => Ok(Py::new(
                py,
                PyClassInitializer::from(self).add_subclass(DtypeSingle),
            )?
            .into_any()),
            DtypeRepr::Array { .. } => Ok(Py::new(
                py,
                PyClassInitializer::from(self).add_subclass(DtypeArray),
            )?
            .into_any()),
            DtypeRepr::Tuple(_) => Ok(Py::new(
                py,
                PyClassInitializer::from(self).add_subclass(DtypeTuple),
            )?
            .into_any()),
        }
    }

    pub(crate) fn single(&self) -> Option<&SingleDtype> {
        match &self.repr {
            DtypeRepr::Single(dtype) => Some(dtype),
            _ => None,
        }
    }

    fn class_name(&self) -> &'static str {
        match self.repr {
            DtypeRepr::Single(_) => "DtypeSingle",
            DtypeRepr::Array { .. } => "DtypeArray",
            DtypeRepr::Tuple(_) => "DtypeTuple",
        }
    }
}

pub(crate) fn extract_dtype(obj: &Bound<'_, PyAny>) -> PyResult<Dtype> {
    if let Ok(dtype) = obj.extract::<PyRef<'_, Dtype>>() {
        return Ok(dtype.clone());
    }
    if let Ok(spec) = obj.extract::<String>() {
        return Dtype::parse_spec(&spec);
    }
    Err(PyTypeError::new_err(
        "dtype must be a Dtype instance or dtype string.",
    ))
}

#[pymethods]
impl Dtype {
    /// Parse a scalar, array or tuple dtype specification.
    ///
    /// :param str spec: The dtype specification.
    /// :return: A concrete dtype instance.
    /// :raises ValueError: if the specification is invalid or does not have a positive fixed length.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(py: Python<'_>, spec: &str) -> PyResult<Py<PyAny>> {
        Self::parse_spec(spec)?.into_python(py)
    }

    /// The number of bits used by one complete value.
    #[getter]
    fn length(&self) -> usize {
        self.length
    }

    /// Encode one Python value as a :class:`Tibs`.
    ///
    /// :param object value: A scalar or structured value matching this dtype.
    /// :return: The encoded bits.
    /// :raises ValueError: if a structured value has the wrong number of items.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("(u8, bool)").pack((15, True))
    ///     Tibs('0b000011111')
    #[pyo3(signature = (value, /), text_signature = "($self, value, /)")]
    fn pack(&self, value: &Bound<'_, PyAny>) -> PyResult<Tibs> {
        Ok(Tibs::from_bv(bv_from_value(self, value)?))
    }

    /// Encode and concatenate Python values as a :class:`Tibs`.
    ///
    /// :param Iterable iterable: Values matching this dtype.
    /// :return: The concatenated encoded bits.
    ///
    /// .. code-block:: pycon
    ///
    ///     >>> Dtype("u8").pack_values([1, 2, 3])
    ///     Tibs('0x010203')
    #[pyo3(signature = (iterable, /), text_signature = "($self, iterable, /)")]
    fn pack_values(&self, py: Python<'_>, iterable: &Bound<'_, PyAny>) -> PyResult<Tibs> {
        Ok(Tibs::from_bv(bv_from_values_iter(py, self, iterable)?))
    }

    /// Decode one complete value from a bit sequence.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :return: One scalar value, or a tuple for an array or tuple dtype.
    /// :raises ValueError: if the selected range is not exactly :attr:`length` bits.
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    fn unpack(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<PyAny>> {
        let bits = bits.extract::<Tibs>()?;
        let (start, end) = validate_slice(bits.len(), start, end)?;
        py_from_value(py, self, &bits.get_slice_unchecked(start, end - start))
    }

    /// Decode a list of complete values from a bit sequence.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :return: A list of decoded values.
    /// :raises ValueError: if the selected range is not a multiple of :attr:`length`.
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    fn unpack_values(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let bits = bits.extract::<Tibs>()?;
        py_values_from_range(py, &bits, self, start, end)
    }

    /// Lazily decode complete values from a bit sequence.
    ///
    /// :param object bits: Anything promotable to :class:`Tibs`.
    /// :param int | None start: Start bit position. Defaults to 0.
    /// :param int | None end: End bit position. Defaults to ``len(bits)``.
    /// :return: An iterator of decoded values.
    /// :raises ValueError: if the selected range is not a multiple of :attr:`length`.
    #[pyo3(signature = (bits, /, start = None, end = None), text_signature = "($self, bits, /, start=None, end=None)")]
    fn unpack_values_iter(
        &self,
        py: Python<'_>,
        bits: &Bound<'_, PyAny>,
        start: Option<isize>,
        end: Option<isize>,
    ) -> PyResult<Py<ValuesIterator>> {
        let bits = bits.extract::<Tibs>()?;
        let (start, end) = validate_slice(bits.len(), start, end)?;
        ValuesIterator::new(py, Py::new(py, bits)?, self.clone(), start, end)
    }

    fn __str__(&self) -> String {
        self.repr.spec()
    }

    fn __repr__(&self) -> String {
        format!("{}('{}')", self.class_name(), self.repr.spec())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.extract::<PyRef<'_, Dtype>>() else {
            return Ok(false);
        };
        Ok(self == &*other)
    }

    fn __hash__(&self) -> isize {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish() as isize;
        if hash == -1 { -2 } else { hash }
    }
}

///     A scalar dtype with a kind, bit length and optional byte order.
///
///     Construct directly from a scalar specification, use
///     :meth:`from_params`, or obtain one from the :class:`Dtype` factory.
///
#[pyclass(module = "tibs", frozen, extends = Dtype, skip_from_py_object)]
pub struct DtypeSingle;

#[pymethods]
impl DtypeSingle {
    /// Parse a scalar dtype specification.
    ///
    /// :param str spec: A specification such as ``"u8"`` or ``"f32_le"``.
    /// :raises ValueError: if ``spec`` describes an array or tuple.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(spec: &str) -> PyResult<PyClassInitializer<Self>> {
        let dtype = Dtype::parse_spec(spec)?;
        if !matches!(dtype.repr, DtypeRepr::Single(_)) {
            return Err(PyValueError::new_err(
                "DtypeSingle requires a scalar dtype specification.",
            ));
        }
        Ok(PyClassInitializer::from(dtype).add_subclass(Self))
    }

    /// Construct a scalar dtype from explicit parameters.
    ///
    /// :param DtypeKind kind: The scalar value kind.
    /// :param int length: The positive bit length.
    /// :param ByteOrder | None byte_order: The byte order. Defaults to unspecified.
    /// :return: A scalar dtype.
    #[classmethod]
    #[pyo3(signature = (kind, length, byte_order = ByteOrder::Unspecified), text_signature = "(cls, kind, length, byte_order=ByteOrder.Unspecified)")]
    fn from_params(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        kind: DtypeKind,
        length: i64,
        byte_order: Option<ByteOrder>,
    ) -> PyResult<Py<Self>> {
        let byte_order = byte_order.unwrap_or(ByteOrder::Unspecified);
        let repr = DtypeRepr::Single(SingleDtype::from_parts(kind, length, byte_order)?);
        Py::new(
            py,
            PyClassInitializer::from(Dtype::from_repr(repr)?).add_subclass(Self),
        )
    }

    /// The scalar value kind.
    #[getter]
    fn kind(slf: PyRef<'_, Self>) -> DtypeKind {
        slf.as_super().single().unwrap().kind
    }

    /// The scalar byte order.
    #[getter]
    fn byte_order(slf: PyRef<'_, Self>) -> ByteOrder {
        slf.as_super().single().unwrap().byte_order
    }
}

///     A fixed positive number of repetitions of another dtype.
///
///     Array values pack from any iterable with exactly :attr:`count` items
///     and unpack to Python tuples.
///
#[pyclass(module = "tibs", frozen, extends = Dtype, skip_from_py_object)]
pub struct DtypeArray;

#[pymethods]
impl DtypeArray {
    /// Parse an array dtype specification.
    ///
    /// :param str spec: A specification such as ``"[u8; 4]"``.
    /// :raises ValueError: if ``spec`` does not describe an array.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(spec: &str) -> PyResult<PyClassInitializer<Self>> {
        let dtype = Dtype::parse_spec(spec)?;
        if !matches!(dtype.repr, DtypeRepr::Array { .. }) {
            return Err(PyValueError::new_err(
                "DtypeArray requires an array dtype specification.",
            ));
        }
        Ok(PyClassInitializer::from(dtype).add_subclass(Self))
    }

    /// Construct an array dtype from its element dtype and count.
    ///
    /// :param Dtype | str dtype: The element dtype.
    /// :param int count: The positive number of elements.
    /// :return: An array dtype.
    #[classmethod]
    #[pyo3(signature = (dtype, count, /), text_signature = "(cls, dtype, count, /)")]
    fn from_params(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        dtype: &Bound<'_, PyAny>,
        count: i64,
    ) -> PyResult<Py<Self>> {
        if count <= 0 {
            return Err(PyValueError::new_err(
                "DtypeArray count must be greater than zero.",
            ));
        }
        let dtype = extract_dtype(dtype)?;
        let repr = DtypeRepr::Array {
            dtype: Box::new(dtype.repr),
            count: count as usize,
        };
        Py::new(
            py,
            PyClassInitializer::from(Dtype::from_repr(repr)?).add_subclass(Self),
        )
    }

    /// The dtype repeated by this array.
    #[getter]
    fn dtype(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let DtypeRepr::Array { dtype, .. } = &slf.as_super().repr else {
            unreachable!()
        };
        Dtype::from_repr((**dtype).clone())?.into_python(py)
    }

    /// The number of values in this array.
    #[getter]
    fn count(slf: PyRef<'_, Self>) -> usize {
        let DtypeRepr::Array { count, .. } = &slf.as_super().repr else {
            unreachable!()
        };
        *count
    }
}

///     An ordered, non-empty tuple of dtypes.
///
///     Tuple values pack from any iterable with exactly one item per field and
///     unpack to Python tuples.
///
#[pyclass(module = "tibs", frozen, extends = Dtype, skip_from_py_object)]
pub struct DtypeTuple;

#[pymethods]
impl DtypeTuple {
    /// Parse a tuple dtype specification.
    ///
    /// :param str spec: A specification such as ``"(u8, u16_le)"``.
    /// :raises ValueError: if ``spec`` does not describe a tuple.
    #[new]
    #[pyo3(signature = (spec, /), text_signature = "(spec, /)")]
    fn py_new(spec: &str) -> PyResult<PyClassInitializer<Self>> {
        let dtype = Dtype::parse_spec(spec)?;
        if !matches!(dtype.repr, DtypeRepr::Tuple(_)) {
            return Err(PyValueError::new_err(
                "DtypeTuple requires a tuple dtype specification.",
            ));
        }
        Ok(PyClassInitializer::from(dtype).add_subclass(Self))
    }

    /// Construct a tuple dtype from its field dtypes.
    ///
    /// :param Iterable dtypes: A non-empty iterable of dtypes or dtype strings.
    /// :return: A tuple dtype.
    #[classmethod]
    #[pyo3(signature = (dtypes, /), text_signature = "(cls, dtypes, /)")]
    fn from_params(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        dtypes: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let reprs = dtypes
            .try_iter()?
            .map(|item| extract_dtype(&item?).map(|dtype| dtype.repr))
            .collect::<PyResult<Vec<_>>>()?;
        if reprs.is_empty() {
            return Err(PyValueError::new_err(
                "DtypeTuple must contain at least one dtype.",
            ));
        }
        Py::new(
            py,
            PyClassInitializer::from(Dtype::from_repr(DtypeRepr::Tuple(reprs))?).add_subclass(Self),
        )
    }

    /// The field dtypes as an immutable Python tuple.
    #[getter]
    fn dtypes(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let DtypeRepr::Tuple(dtypes) = &slf.as_super().repr else {
            unreachable!()
        };
        let objects = dtypes
            .iter()
            .cloned()
            .map(|repr| Dtype::from_repr(repr)?.into_python(py))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, objects)?.unbind())
    }
}
